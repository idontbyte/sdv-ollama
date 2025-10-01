using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input; // Keys
using StardewModdingAPI;
using StardewModdingAPI.Events;
using StardewValley;
using StardewValley.Menus;
using StardewValley.BellsAndWhistles; // SpriteText

namespace LLMAllDialogue
{
    public sealed class ModConfig
    {
        public string Endpoint { get; set; } = "http://localhost:11434/api/generate"; // Ollama /api/generate
        public string Model { get; set; } = "gemma3:270m";
        public string ApiKey { get; set; } = "";
        public double Temperature { get; set; } = 0.7;
        public int MaxChars { get; set; } = 220;
        public int MaxTurns { get; set; } = 8; // (player+NPC) pairs kept in rolling context

        // You can still change the base system prompt; strict rules get added automatically.
        public string SystemPrompt { get; set; } =
@"You are roleplaying as an NPC from Stardew Valley. Stay wholesome, concise, and in-universe.
No explicit content. Avoid modern internet slang.";

        public bool EnableForCutscenes { get; set; } = true;
        public int RequestTimeoutSeconds { get; set; } = 12;
        public int InputCharLimit { get; set; } = 240; // cap input length on send

        // Extras
        public bool Debug { get; set; } = true;            // extra logging
        public bool StartDayAtNine { get; set; } = true;   // set time to 9:00 on day start/load
        public int LoreLines { get; set; } = 6;            // how many vanilla lines to feed as 'lore'

        // Prompt budget (approx token cap). 6000 chars ~ 1500 tokens-ish.
        public int MaxPromptChars { get; set; } = 6000;
    }

    public interface ILlmClient
    {
        Task<string> GenerateAsync(string systemPrompt, string userPrompt, double temperature, int maxChars, CancellationToken ct);
    }

    /// <summary>JSON client for Ollama /api/generate with verbose debug logging.</summary>
    public sealed class HttpJsonLlmClient : ILlmClient
    {
        private readonly HttpClient _http = new();
        private readonly string _endpoint, _model;
        private readonly IMonitor _monitor;
        private readonly bool _debug;

        public HttpJsonLlmClient(IMonitor monitor, string endpoint, string model, string apiKey, bool debug)
        {
            _monitor = monitor;
            _endpoint = endpoint?.TrimEnd('/') ?? "";
            _model = model ?? "";
            _debug = debug;

            if (!string.IsNullOrWhiteSpace(apiKey))
                _http.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
        }

        public async Task<string> GenerateAsync(string systemPrompt, string userPrompt, double temperature, int maxChars, CancellationToken ct)
        {
            // Log exactly what we're sending (sanitized/truncated)
            if (_debug)
            {
                _monitor.Log($"[LLM] SYSTEM ▶\n{Truncate(systemPrompt, 1200)}", LogLevel.Trace);
                _monitor.Log($"[LLM] USER ▶\n{Truncate(userPrompt, 2000)}", LogLevel.Trace);
            }

            var payload = new
            {
                model = _model,
                prompt = $"{systemPrompt}\n\n{userPrompt}\n\n",
                options = new { temperature = temperature }
            };

            var json = JsonSerializer.Serialize(payload);
            using var req = new HttpRequestMessage(HttpMethod.Post, _endpoint)
            {
                Content = new StringContent(json, Encoding.UTF8, "application/json")
            };

            if (_debug)
            {
                _monitor.Log($"[LLM] POST {_endpoint} | model={_model} temp={temperature} maxChars={maxChars}", LogLevel.Trace);
                _monitor.Log($"[LLM] JSON payload preview ▶\n{Truncate(json, 1400)}", LogLevel.Trace);
            }

            HttpResponseMessage res = null!;
            string body = "";
            try
            {
                res = await _http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
                body = await res.Content.ReadAsStringAsync(ct);

                if (!res.IsSuccessStatusCode)
                    throw new Exception($"HTTP {(int)res.StatusCode} {res.ReasonPhrase} :: {Truncate(body, 600)}");

                if (_debug)
                    _monitor.Log($"[LLM] {(int)res.StatusCode} OK, body_len={body.Length}", LogLevel.Trace);
            }
            catch (Exception ex)
            {
                _monitor.Log($"[LLM] Request failed: {ex.Message}", LogLevel.Warn);
                return "(looks confused)";
            }

            // Parse Ollama streaming NDJSON
            if (body.Contains("\"response\"") && body.Contains("\n"))
            {
                var sb = new StringBuilder();
                foreach (var line in body.Split('\n'))
                {
                    var t = line.Trim();
                    if (!t.StartsWith("{")) continue;
                    try
                    {
                        using var doc = JsonDocument.Parse(t);
                        if (doc.RootElement.TryGetProperty("response", out var resp))
                            sb.Append(resp.GetString());
                    }
                    catch { /* ignore */ }
                }
                var outText = Truncate(Clean(sb.ToString()), maxChars);
                if (_debug) _monitor.Log($"[LLM] parsed streamed text ▶ \"{Truncate(outText, 200)}\"", LogLevel.Trace);
                return outText;
            }

            // Fallback single JSON
            try
            {
                using var doc = JsonDocument.Parse(body);
                if (doc.RootElement.TryGetProperty("response", out var resp2))
                {
                    var outText = Truncate(Clean(resp2.GetString() ?? ""), maxChars);
                    if (_debug) _monitor.Log($"[LLM] parsed single response ▶ \"{Truncate(outText, 200)}\"", LogLevel.Trace);
                    return outText;
                }
            }
            catch
            {
                // ignore, raw fallback below
            }

            var fallback = Truncate(Clean(body), maxChars);
            if (_debug) _monitor.Log($"[LLM] raw fallback ▶ \"{Truncate(fallback, 200)}\"", LogLevel.Trace);
            return fallback;
        }

        private static string Clean(string s)
        {
            s = Regex.Replace(s, @"```[\s\S]*?```", "").Trim();
            s = s.Replace("\r", "");
            return s;
        }

        private static string Truncate(string s, int max) => (max <= 0 || s.Length <= max) ? s : s[..max] + "…";
    }

    public class ModEntry : Mod
    {
        private ModConfig _config = null!;
        private ILlmClient _llm = null!;

        private readonly ConcurrentDictionary<string, List<(string who, string text)>> _threads = new();
        private readonly ConcurrentDictionary<string, List<string>> _loreCache = new(); // per-NPC vanilla snippets

        public override void Entry(IModHelper helper)
        {
            _config = helper.ReadConfig<ModConfig>();
            _llm = new HttpJsonLlmClient(Monitor, _config.Endpoint, _config.Model, _config.ApiKey, _config.Debug);

            helper.Events.Display.MenuChanged += OnMenuChanged;
            helper.Events.GameLoop.SaveLoaded += OnSaveLoaded;
            helper.Events.GameLoop.DayStarted += OnDayStarted;

            Monitor.Log("LLM chat loaded.", LogLevel.Info);
        }

        private void OnSaveLoaded(object? sender, SaveLoadedEventArgs e)
        {
            if (_config.StartDayAtNine)
            {
                Game1.timeOfDay = 900;
                Monitor.Log("[Time] Set to 9:00 AM (on save load).", LogLevel.Info);
            }
        }

        private void OnDayStarted(object? sender, DayStartedEventArgs e)
        {
            if (_config.StartDayAtNine)
            {
                Game1.timeOfDay = 900;
                Monitor.Log("[Time] Set to 9:00 AM (new day).", LogLevel.Info);
            }
        }

        private void OnMenuChanged(object? sender, MenuChangedEventArgs e)
        {
            if (e.NewMenu is not DialogueBox dbox)
                return;

            var npc = dbox.characterDialogue?.speaker;
            if (npc is null) return;
            if (!_config.EnableForCutscenes && Game1.eventUp) return;

            // opening line for grounding
            var vanilla = dbox.characterDialogue?.getCurrentDialogue() ?? "";

            if (ReferenceEquals(Game1.activeClickableMenu, dbox))
                Game1.activeClickableMenu = null;

            var chat = new ChatWithNpcMenu(npc, vanilla, _config.InputCharLimit, SendAsync);
            Game1.activeClickableMenu = chat;

            var key = NpcKey(npc);
            var list = _threads.GetOrAdd(key, _ => new());
            list.Clear();
            if (!string.IsNullOrWhiteSpace(vanilla))
                list.Add((npc.Name, vanilla));

            // cache lore once per NPC
            _loreCache.TryAdd(npc.Name, TryGetNpcLore(npc, _config.LoreLines));

            if (_config.Debug)
                Monitor.Log($"[Chat] Opened with {npc.Name} @ {Game1.currentLocation?.Name}", LogLevel.Trace);
        }

        private string NpcKey(NPC npc) => npc.Name;

        private async Task<string> SendAsync(NPC npc, string playerText, string visibleLastNpcLine, CancellationToken externalCt)
        {
            var key = NpcKey(npc);
            var thread = _threads.GetOrAdd(key, _ => new());
            thread.Add((Game1.player?.Name ?? "Farmer", playerText));

            if (_config.Debug)
                Monitor.Log($"[Chat] Player → {npc.Name}: \"{playerText}\"", LogLevel.Trace);

            // Build a prompt and trim history/lore to fit char budget
            var userPrompt = BuildPromptWithBudget(npc, thread, visibleLastNpcLine, _config.MaxPromptChars);

            if (_config.Debug)
                Monitor.Log($"[LLM] USER prompt (final, <= {_config.MaxPromptChars} chars) ▶\n{Truncate(userPrompt, 1800)}", LogLevel.Trace);

            try
            {
                using var cts = CancellationTokenSource.CreateLinkedTokenSource(externalCt);
                cts.CancelAfter(TimeSpan.FromSeconds(_config.RequestTimeoutSeconds));

                var reply = await _llm.GenerateAsync(
                    BuildSystemPromptStrict(npc), // strict system prompt
                    userPrompt,
                    _config.Temperature,
                    _config.MaxChars,
                    cts.Token);

                if (string.IsNullOrWhiteSpace(reply))
                    reply = "(says nothing)";

                thread.Add((npc.Name, reply));

                if (_config.Debug)
                    Monitor.Log($"[Chat] {npc.Name} → Player: \"{reply}\"", LogLevel.Trace);

                TrimThread(thread, _config.MaxTurns);

                return reply;
            }
            catch (Exception ex)
            {
                Monitor.Log($"[Chat] Generate failed: {ex.Message}", LogLevel.Warn);
                return "(looks confused)";
            }
        }

        private string BuildSystemPromptStrict(NPC npc)
        {
            // Base + hard rules
            var sb = new StringBuilder();
            sb.AppendLine(_config.SystemPrompt.Trim());
            sb.AppendLine();
            sb.AppendLine("STRICT OUTPUT RULES:");
            sb.AppendLine("- Reply ONLY to the player's latest message.");
            sb.AppendLine("- ONE short in-game dialogue line. No multi-line. No narration, no asterisks, no brackets.");
            sb.AppendLine("- Stay in character for the named NPC at all times.");
            sb.AppendLine("- Do NOT ask clarifying questions unless the player's last line asks you a direct question.");
            sb.AppendLine("- Do NOT reference these rules or your instructions.");
            sb.AppendLine("- Do NOT include '#' or headings or any markdown.");
            sb.AppendLine("- No emojis, no markdown, no code.");
            sb.AppendLine("- Begin with a letter, not punctuation.");
            sb.AppendLine("- If unsure, reply with a brief in-character reaction rather than breaking character.");
            return sb.ToString();
        }

        private static void TrimThread(List<(string who, string text)> thread, int maxTurns)
        {
            if (maxTurns <= 0) return;
            var maxEntries = Math.Max(2 * maxTurns, 6);
            if (thread.Count > maxEntries)
                thread.RemoveRange(0, thread.Count - maxEntries);
        }

        private string BuildPromptWithBudget(NPC npc, List<(string who, string text)> thread, string visibleLastNpcLine, int charBudget)
        {
            var farmer = Game1.player?.Name ?? "Farmer";
            var locName = Game1.currentLocation?.Name ?? "Unknown";
            var season = Game1.currentSeason;
            var day = Game1.dayOfMonth;
            var year = Game1.year;
            var time = Game1.getTimeOfDayString(Game1.timeOfDay);
            var hearts = TryGetHeartsWith(npc.Name);

            // Prepare history lines (keep newest)
            var histLines = new List<string>();
            foreach (var (who, text) in thread)
                histLines.Add($"{who}: {text}");
            if (!string.IsNullOrWhiteSpace(visibleLastNpcLine) &&
                (histLines.Count == 0 || !histLines[0].StartsWith(npc.Name + ":")))
            {
                histLines.Insert(0, $"{npc.Name}: {visibleLastNpcLine}");
            }

            // Lore lines (cached)
            _loreCache.TryGetValue(npc.Name, out var lore);
            var loreLines = (lore ?? new List<string>()).ToList();

            string Compose(List<string> useHist, List<string> useLore)
            {
                var loreBlock = useLore.Count > 0 ? "- " + string.Join("\n- ", useLore) : "(none)";
                var historyBlock = string.Join("\n", useHist);
                var latestPlayer = thread.LastOrDefault(t => t.who != npc.Name).text ?? "";
                return
$@"You are {npc.Name}, an NPC in Stardew Valley, talking to {farmer}.

World State:
- Location: {locName}
- Date: {season} {day}, Year {year} at {time}
- Hearts with player: {hearts}

{npc.Name}'s vanilla personality snippets (from the game):
{loreBlock}

Recent conversation (most recent last):
{historyBlock}

Player's latest message (respond ONLY to this):
""{latestPlayer}""

Reply with ONE short in-universe line, keeping {npc.Name}'s voice and mood consistent with the snippets above.";
            }

            // Start with full history (newest last) but we'll trim oldest first
            var hist = histLines.ToList();
            var loreUse = loreLines.ToList();

            string prompt = Compose(hist, loreUse);
            // Trim until within char budget
            while (prompt.Length > charBudget)
            {
                // Remove oldest history first
                if (hist.Count > 1)
                    hist.RemoveAt(0);
                else if (loreUse.Count > 0)
                    loreUse.RemoveAt(loreUse.Count - 1); // trim least important lore (from the end)
                else
                    break; // nothing left to trim
                prompt = Compose(hist, loreUse);
            }

            if (_config.Debug && prompt.Length > charBudget)
                Monitor.Log($"[LLM] Prompt still over budget after trimming: {prompt.Length} chars (budget {charBudget})", LogLevel.Trace);
            if (_config.Debug)
                Monitor.Log($"[LLM] Using history lines: {hist.Count}, lore lines: {loreUse.Count}", LogLevel.Trace);

            return prompt;
        }

        private int TryGetHeartsWith(string npcName)
        {
            if (Game1.player?.friendshipData is null) return 0;
            if (Game1.player.friendshipData.TryGetValue(npcName, out var fs))
                return fs.Points / 250;
            return 0;
        }

        private List<string> TryGetNpcLore(NPC npc, int maxLines)
        {
            var lines = new List<string>();
            try
            {
                // Load vanilla dialogue page for this NPC (vanilla only)
                var path = $"Characters/Dialogue/{npc.Name}";
                var dict = Helper.GameContent.Load<Dictionary<string, string>>(path);
                if (dict != null && dict.Count > 0)
                {
                    // Prefer some common keys, then fill with others.
                    string[] prefer =
                    {
                        "Introduction","Mon","Tue","Wed","Thu","Fri","Sat","Sun",
                        "spring_1","summer_1","fall_1","winter_1","Rainy_Day_0","Rainy_Day_1"
                    };

                    IEnumerable<KeyValuePair<string,string>> ordered =
                        dict.Where(kv => !string.IsNullOrWhiteSpace(kv.Value));

                    var preferred = ordered.Where(kv => prefer.Contains(kv.Key)).ToList();
                    var rest = ordered.Where(kv => !prefer.Contains(kv.Key)).ToList();

                    foreach (var kv in preferred.Concat(rest))
                    {
                        var v = SanitizeDialogue(kv.Value);
                        if (!string.IsNullOrWhiteSpace(v))
                            lines.Add(Truncate(v, 140));
                        if (lines.Count >= Math.Max(1, maxLines)) break;
                    }
                }
            }
            catch
            {
                // ignore missing/unsupported NPCs
            }
            return lines;
        }

        private static string SanitizeDialogue(string s)
        {
            if (string.IsNullOrEmpty(s)) return s;
            // Strip control codes like $s, #$b, ^, etc. and stray '#'
            s = Regex.Replace(s, @"[#$][a-zA-Z0-9_]+", " ");
            s = s.Replace("^", " ").Replace("@", " ").Replace("%", " ");
            s = s.Replace("#", " "); // extra guard against heading-like artifacts
            s = Regex.Replace(s, @"\s+", " ").Trim();
            return s;
        }

        private static string Truncate(string s, int max) => (max <= 0 || s.Length <= max) ? s : s[..max] + "…";
    }

    /// <summary>
    /// Chat UI with NPC name, last NPC line, and a TextBox to type your message.
    /// </summary>
    internal class ChatWithNpcMenu : IClickableMenu
    {
        private readonly NPC _npc;
        private readonly Func<NPC, string, string, CancellationToken, Task<string>> _sender;

        private string _npcLine;          // what we show in the bubble
        private readonly TextBox _input;  // player's input
        private readonly ClickableTextureComponent _sendBtn;
        private bool _busy;
        private string _status = "";

        private readonly Rectangle _dialogueArea;
        private readonly Rectangle _inputArea;

        private readonly int _inputCharLimit;

        public ChatWithNpcMenu(NPC npc, string openingNpcLine, int inputCharLimit,
            Func<NPC, string, string, CancellationToken, Task<string>> sender)
        {
            _npc = npc;
            _npcLine = string.IsNullOrWhiteSpace(openingNpcLine) ? "(…)" : openingNpcLine;
            _sender = sender;
            _inputCharLimit = Math.Max(1, inputCharLimit);

            width = Math.Min(1100, Game1.uiViewport.Width - 128);
            height = Math.Min(380, Game1.uiViewport.Height - 160);
            xPositionOnScreen = Game1.uiViewport.Width / 2 - width / 2;
            yPositionOnScreen = Game1.uiViewport.Height - height - 48;

            _dialogueArea = new Rectangle(xPositionOnScreen + 64, yPositionOnScreen + 64, width - 128, 160);
            _inputArea = new Rectangle(xPositionOnScreen + 64, yPositionOnScreen + height - 100, width - 128 - 72, 64);

            _input = new TextBox(null, null, Game1.dialogueFont, Game1.textColor)
            {
                X = _inputArea.X + 8,
                Y = _inputArea.Y + 8,
                Width = _inputArea.Width - 16,
                Height = _inputArea.Height - 16
            };
            Game1.keyboardDispatcher.Subscriber = _input;

            _sendBtn = new ClickableTextureComponent(
                new Rectangle(_inputArea.Right + 8, _inputArea.Y + 8, 56, _inputArea.Height - 16),
                Game1.mouseCursors, new Rectangle(128, 256, 64, 64), 0.7f)
            {
                hoverText = "Send"
            };
        }

        public override void receiveKeyPress(Keys key)
        {
            bool typing = Game1.keyboardDispatcher.Subscriber == _input;

            if (typing)
            {
                if (key == Keys.Enter)
                {
                    _ = TrySendAsync();
                    Game1.playSound("smallSelect");
                    return;
                }
                if (key == Keys.Escape)
                {
                    exitThisMenu();
                    return;
                }
                // swallow other keys; TextBox gets them via keyboard dispatcher
                return;
            }

            // Not typing: allow closing with Escape or menu button
            if (key == Keys.Escape || Game1.options.menuButton.Contains(new InputButton(key)))
            {
                exitThisMenu();
                return;
            }

            base.receiveKeyPress(key);
        }

        public override void receiveLeftClick(int x, int y, bool playSound = true)
        {
            if (_sendBtn.containsPoint(x, y))
            {
                _ = TrySendAsync();
                if (playSound) Game1.playSound("smallSelect");
            }
            base.receiveLeftClick(x, y, playSound);
        }

        private async Task TrySendAsync()
        {
            if (_busy) return;

            var text = _input.Text?.Trim() ?? "";
            if (string.IsNullOrEmpty(text))
            {
                Game1.playSound("cancel");
                return;
            }

            // Clamp to configured char limit
            if (text.Length > _inputCharLimit)
                text = text.Substring(0, _inputCharLimit);

            _busy = true;
            _status = "(thinking…)";

            var visibleNpc = _npcLine;
            _input.Text = "";

            string reply;
            try
            {
                using var cts = new CancellationTokenSource();
                reply = await _sender(_npc, text, visibleNpc, cts.Token);
            }
            catch
            {
                reply = "(no response)";
            }

            _npcLine = reply;
            _status = "";
            _busy = false;
        }

        public override void performHoverAction(int x, int y)
        {
            _sendBtn.tryHover(x, y);
            base.performHoverAction(x, y);
        }

        public override void draw(SpriteBatch b)
        {
            IClickableMenu.drawTextureBox(b, xPositionOnScreen, yPositionOnScreen, width, height, Color.White);

            // NPC name (portrait-free for compatibility)
            SpriteText.drawString(b, _npc.displayName, xPositionOnScreen + 32, yPositionOnScreen + 24);

            // Dialogue bubble area
            var textRect = _dialogueArea;
            IClickableMenu.drawTextureBox(b, textRect.X - 16, textRect.Y - 12, textRect.Width + 32, textRect.Height + 24, Color.White);
            var wrapped = Game1.parseText(_npcLine, Game1.dialogueFont, textRect.Width - 24);
            Utility.drawTextWithShadow(b, wrapped, Game1.dialogueFont, new Vector2(textRect.X + 12, textRect.Y + 12), Game1.textColor);

            if (!string.IsNullOrEmpty(_status))
                Utility.drawTextWithShadow(b, _status, Game1.smallFont, new Vector2(textRect.Right - 160, textRect.Bottom + 8), Game1.textColor);

            // Input area + box
            IClickableMenu.drawTextureBox(b, _inputArea.X - 8, _inputArea.Y - 8, _inputArea.Width + 16, _inputArea.Height + 16, Color.White);
            _input.Draw(b);

            // Send button
            _sendBtn.draw(b);

            drawMouse(b);
        }

        public override void gameWindowSizeChanged(Rectangle oldBounds, Rectangle newBounds)
        {
            exitThisMenu();
        }

        protected override void cleanupBeforeExit()
        {
            if (Game1.keyboardDispatcher.Subscriber == _input)
                Game1.keyboardDispatcher.Subscriber = null;
            base.cleanupBeforeExit();
        }
    }
}
