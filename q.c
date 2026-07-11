#include <curl/curl.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

#define DEFAULT_MODEL "gpt-5.5"
#define DEFAULT_LLM_SERVER "127.0.0.1"
#define DEFAULT_LLM_PORT "8080"
#define DEFAULT_LLM_TIMEOUT -1L
#define DEFAULT_LLM_TURN_LIMIT 30

#ifndef Q_COLOR_RESET
#define Q_COLOR_RESET "\033[0m"
#endif
#ifndef Q_COLOR_LLM_TEXT
#define Q_COLOR_LLM_TEXT "\033[34m"
#endif
#ifndef Q_COLOR_CODE_EMULATOR
#define Q_COLOR_CODE_EMULATOR "\033[97;100m"
#endif
#ifndef Q_COLOR_CODE_TTY
#define Q_COLOR_CODE_TTY "\033[97m"
#endif
#ifndef Q_COLOR_PROMPT_LINE_NO
#define Q_COLOR_PROMPT_LINE_NO "\033[31;2m"
#endif
#ifndef Q_COLOR_EXIT_OK
#define Q_COLOR_EXIT_OK "\033[32m"
#endif
#ifndef Q_COLOR_EXIT_FAIL
#define Q_COLOR_EXIT_FAIL "\033[31m"
#endif
#ifndef Q_COLOR_MENU_SELECTED
#define Q_COLOR_MENU_SELECTED "\033[7m"
#endif
#ifndef Q_COLOR_TOOL_CONFIRM
#define Q_COLOR_TOOL_CONFIRM "\033[38;5;166m"
#endif
#ifndef Q_COLOR_EXEC_BLOCK_LABEL
#define Q_COLOR_EXEC_BLOCK_LABEL "\033[34m"
#endif

static const char *SYSTEM_PROMPT =
    "System Prompt - Linux Sysadmin Assistant\n"
    "You are a senior Linux system administrator with 15+ years of hands-on experience. "
    "Your primary environment is Linux Mint (and its Ubuntu/Debian base), though you are equally fluent across all major distributions.\n"
    "Persona & Expertise\n"
    "Deep expertise in the Linux CLI, shell scripting (bash/zsh), cron, systemd, networking, package management (apt, dpkg, snap, flatpak), file systems, permissions, users/groups, firewalls (ufw, iptables), and performance tuning.\n"
    "Familiar with desktop environments (Cinnamon, MATE, XFCE) and their configuration.\n"
    "Comfortable at the kernel level: modules, dmesg, grub, initrd, and boot troubleshooting.\n"
    "Proficient with common tools: rsync, ssh, tmux, vim, grep, awk, sed, find, netstat/ss, htop, journalctl, lsof, strace, and more.\n"
    "Response Style\n"
    "Concise. No filler, no hand-holding, no unnecessary explanation.\n"
    "Do not overthink. Perform only the minimum checks needed to avoid a wrong or dangerous answer, then answer directly.\n"
    "If checks are needed, limit them to the most relevant details and do not narrate the checking process unless asked.\n"
    "Answer only what was asked.\n"
    "If context is missing, ask one targeted question.\n"
    "Assume the user knows Linux basics - skip obvious caveats unless genuinely dangerous.\n"
    "For dangerous operations (data loss, system-breaking), add a single short warning line - nothing more.\n"
    "Formatting Rules\n"
    "Prose answers are plain and brief - typically 1-4 sentences.\n"
    "Any shell command, pipeline, one-liner, or shell script MUST be inside a fenced code block tagged with the actual shell language: sh for POSIX shell, bash for bash, or zsh for zsh.\n"
    "Configuration files, service units, application configs, and non-shell file contents MUST use fenced code blocks with no language word after the opening backticks.\n"
    "Never output executable shell text outside fenced code blocks.\n"
    "If the user specifically asks you to read a file or write a file, use the provided read_file or write_file tool call instead of giving code blocks, samples, snippets, or shell commands.\n"
    "Use the provided get_time tool at the start of every interaction to obtain the current local and UTC time before answering.\n"
    "If MCP tools are available and they are relevant, use them when they can answer more directly or when local information is insufficient.\n"
    "If a requested read_file target in the current working directory does not exist, tell the user the file is not present after the tool result says so.\n"
    "Fence markers have no indentation.\n"
    "If multiple steps are needed, use a minimal numbered list.\n"
    "No decorative headers, no excessive markdown.\n"
    "Only when asked for explanation you do provide it.";

struct buffer {
    char *data;
    size_t len;
};

struct stream_state {
    struct buffer raw;
    struct buffer line;
    char *used_model;
    long input_tokens;
    long output_tokens;
    long total_tokens;
    long context_used_tokens;
    int printed;
    int think_loud;
    int thinking_indicator;
    int thinking_frame;
    int turn_number;
    int waiting_indicator;
    int answer_started;
    int received_first_byte;
    int quiet;
    int answer_color;
    const char *code_color;
    int answer_color_started;
    int answer_color_code;
    int in_code_fence;
    int pending_backticks;
    struct buffer answer_text;
    struct buffer tool_calls;
    char *tool_call_id;
    char *tool_name;
    char *tool_arguments;
};

struct command_result {
    char *command;
    char *direct_tty_name;
    char *output;
    int exit_code;
    int valid;
    int direct_tty;
};

struct code_blocks {
    char **items;
    size_t len;
    size_t cap;
};

struct chat_message {
    char *role;
    char *content;
};

struct chat_history {
    struct chat_message *items;
    size_t len;
    size_t cap;
    char **input_items;
    size_t input_len;
    size_t input_cap;
    int enabled;
    int include_context;
    int persist;
    const char *path;
};

struct shell_aliases {
    char **names;
    char **values;
    size_t len;
    size_t cap;
};

struct mcp_server {
    char *name;
    char *command;
    int alive;
    size_t tool_count;
};

struct mcp_tool {
    char *server;
    char *tool;
    char *function;
    char *description;
    char *schema;
};

struct mcp_registry {
    struct mcp_server *servers;
    size_t server_len;
    size_t server_cap;
    struct mcp_tool *tools;
    size_t tool_len;
    size_t tool_cap;
};

static struct mcp_registry mcp_registry = {0};

static volatile sig_atomic_t shutdown_signal = 0;
static volatile sig_atomic_t request_interrupted = 0;
static volatile sig_atomic_t request_interrupt_signal = 0;
static struct termios saved_termios;
static int saved_termios_valid = 0;
static int terminal_raw_active = 0;
static int terminal_cursor_hidden = 0;
static long llm_timeout_seconds = DEFAULT_LLM_TIMEOUT;
static long llm_turn_limit = DEFAULT_LLM_TURN_LIMIT;
static char *api_logging_path = NULL;
static char *system_prompt_text = NULL;
static char *system_prompt_path = NULL;

enum api_logging_mode {
    API_LOGGING_NONE = 0,
    API_LOGGING_QUERY = 1,
    API_LOGGING_RESPONSE = 2,
    API_LOGGING_BOTH = 3
};

static char *get_json_string(const char *json, const char *key);
static char *execute_mcp_tool_call(const char *function, const char *arguments);
static char *read_text_file(const char *path);
static int write_text_file(const char *path, const char *text);
static const char *api_logging_mode_name(int mode);
static void print_llm_timeout_value(FILE *out, long value);
static char *shell_quote_word(const char *s);
static int clipboard_available(void);

static void hide_terminal_cursor(void) {
    if (!terminal_cursor_hidden) {
        fputs("\033[?25l", stderr);
        fflush(stderr);
        terminal_cursor_hidden = 1;
    }
}

static void show_terminal_cursor(void) {
    if (terminal_cursor_hidden) {
        fputs("\033[?25h", stderr);
        fflush(stderr);
        terminal_cursor_hidden = 0;
    }
}

static void restore_terminal_if_needed(void) {
    show_terminal_cursor();
    if (terminal_raw_active && saved_termios_valid) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &saved_termios);
        terminal_raw_active = 0;
    }
}

static void enable_raw_from_saved_if_needed(void) {
    if (!saved_termios_valid || terminal_raw_active) {
        return;
    }
    struct termios raw = saved_termios;
    raw.c_lflag &= (tcflag_t)~(ECHO | ICANON | IEXTEN);
    raw.c_iflag &= (tcflag_t)~(IXON | ICRNL);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) == 0) {
        terminal_raw_active = 1;
    }
}

static void handle_shutdown_signal(int sig) {
    shutdown_signal = sig;
}

static void handle_interrupt_signal(int sig) {
    request_interrupted = 1;
    request_interrupt_signal = sig;
}

static void install_shutdown_handlers(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_shutdown_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGHUP, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGQUIT, &sa, NULL);

    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_interrupt_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
}

static double now_seconds(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static int append_bytes(struct buffer *buf, const char *data, size_t n) {
    char *next = realloc(buf->data, buf->len + n + 1);
    if (!next) {
        return 0;
    }
    buf->data = next;
    memcpy(buf->data + buf->len, data, n);
    buf->len += n;
    buf->data[buf->len] = '\0';
    return 1;
}

static char *join_args(int argc, char **argv, int first_arg) {
    size_t len = 0;
    for (int i = first_arg; i < argc; i++) {
        len += strlen(argv[i]) + (i > first_arg ? 1 : 0);
    }

    char *out = malloc(len + 1);
    if (!out) {
        return NULL;
    }

    out[0] = '\0';
    for (int i = first_arg; i < argc; i++) {
        if (i > first_arg) {
            strcat(out, " ");
        }
        strcat(out, argv[i]);
    }
    return out;
}

static char *json_escape(const char *s) {
    size_t len = 2;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
            case '"':
            case '\\':
            case '\b':
            case '\f':
            case '\n':
            case '\r':
            case '\t':
                len += 2;
                break;
            default:
                len += *p < 0x20 ? 6 : 1;
                break;
        }
    }

    char *out = malloc(len + 1);
    if (!out) {
        return NULL;
    }

    char *q = out;
    *q++ = '"';
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
            case '"': *q++ = '\\'; *q++ = '"'; break;
            case '\\': *q++ = '\\'; *q++ = '\\'; break;
            case '\b': *q++ = '\\'; *q++ = 'b'; break;
            case '\f': *q++ = '\\'; *q++ = 'f'; break;
            case '\n': *q++ = '\\'; *q++ = 'n'; break;
            case '\r': *q++ = '\\'; *q++ = 'r'; break;
            case '\t': *q++ = '\\'; *q++ = 't'; break;
            default:
                if (*p < 0x20) {
                    snprintf(q, 7, "\\u%04x", *p);
                    q += 6;
                } else {
                    *q++ = (char)*p;
                }
                break;
        }
    }
    *q++ = '"';
    *q = '\0';
    return out;
}

static void chat_history_free(struct chat_history *history) {
    for (size_t i = 0; i < history->len; i++) {
        free(history->items[i].role);
        free(history->items[i].content);
    }
    for (size_t i = 0; i < history->input_len; i++) {
        free(history->input_items[i]);
    }
    free(history->items);
    free(history->input_items);
    history->items = NULL;
    history->input_items = NULL;
    history->len = 0;
    history->cap = 0;
    history->input_len = 0;
    history->input_cap = 0;
}

static int chat_history_add(struct chat_history *history, const char *role, const char *content) {
    if (!history || !history->enabled) {
        return 1;
    }
    if (history->len == history->cap) {
        size_t next_cap = history->cap ? history->cap * 2 : 16;
        struct chat_message *next = realloc(history->items, next_cap * sizeof(*next));
        if (!next) {
            return 0;
        }
        history->items = next;
        history->cap = next_cap;
    }

    history->items[history->len].role = strdup(role);
    history->items[history->len].content = strdup(content);
    if (!history->items[history->len].role || !history->items[history->len].content) {
        free(history->items[history->len].role);
        free(history->items[history->len].content);
        return 0;
    }
    history->len++;
    return 1;
}

static int chat_history_append_file(const struct chat_history *history, const char *role, const char *content) {
    if (!history || !history->persist || !history->path || !*history->path) {
        return 1;
    }

    FILE *fp = fopen(history->path, "a");
    if (!fp) {
        perror("context fopen");
        return 0;
    }

    char *r = json_escape(role);
    char *c = json_escape(content);
    if (!r || !c) {
        free(r);
        free(c);
        fclose(fp);
        return 0;
    }

    int ok = fprintf(fp, "{\"role\":%s,\"content\":%s}\n", r, c) >= 0;
    free(r);
    free(c);
    if (fclose(fp) != 0) {
        perror("context fclose");
        ok = 0;
    }
    return ok;
}

static int chat_history_add_input(struct chat_history *history, const char *line) {
    if (!history || !history->enabled || !line || !*line) {
        return 1;
    }
    if (history->input_len == history->input_cap) {
        size_t next_cap = history->input_cap ? history->input_cap * 2 : 32;
        char **next = realloc(history->input_items, next_cap * sizeof(*next));
        if (!next) {
            return 0;
        }
        history->input_items = next;
        history->input_cap = next_cap;
    }
    history->input_items[history->input_len] = strdup(line);
    if (!history->input_items[history->input_len]) {
        return 0;
    }
    history->input_len++;
    return 1;
}

static int chat_history_append_input_file(const struct chat_history *history, const char *line) {
    if (!history || !history->persist || !history->path || !*history->path || !line || !*line) {
        return 1;
    }

    FILE *fp = fopen(history->path, "a");
    if (!fp) {
        perror("session input fopen");
        return 0;
    }

    char *c = json_escape(line);
    if (!c) {
        fclose(fp);
        return 0;
    }

    int ok = fprintf(fp, "{\"history\":%s}\n", c) >= 0;
    free(c);
    if (fclose(fp) != 0) {
        perror("session input fclose");
        ok = 0;
    }
    return ok;
}

static int chat_history_record_input(struct chat_history *history, const char *line) {
    if (!chat_history_add_input(history, line)) {
        return 0;
    }
    return chat_history_append_input_file(history, line);
}

static int chat_history_append_event(const struct chat_history *history, const char *event, int sig) {
    if (!history || !history->persist || !history->path || !*history->path) {
        return 1;
    }

    FILE *fp = fopen(history->path, "a");
    if (!fp) {
        perror("session event fopen");
        return 0;
    }

    time_t now = time(NULL);
    struct tm tm_now;
    localtime_r(&now, &tm_now);
    char when[32];
    strftime(when, sizeof(when), "%Y-%m-%dT%H:%M:%S%z", &tm_now);

    char *e = json_escape(event ? event : "");
    char *w = json_escape(when);
    if (!e || !w) {
        free(e);
        free(w);
        fclose(fp);
        return 0;
    }

    int ok = fprintf(fp, "{\"event\":%s,\"signal\":%d,\"time\":%s}\n", e, sig, w) >= 0;
    free(e);
    free(w);
    if (fclose(fp) != 0) {
        perror("session event fclose");
        ok = 0;
    }
    return ok;
}

static int chat_history_load(struct chat_history *history) {
    if (!history || !history->enabled || !history->path || !*history->path) {
        return 1;
    }

    FILE *fp = fopen(history->path, "r");
    if (!fp) {
        return 1;
    }

    char *line = NULL;
    size_t cap = 0;
    ssize_t n;
    int ok = 1;
    while ((n = getline(&line, &cap, fp)) >= 0) {
        (void)n;
        char *role = get_json_string(line, "role");
        char *content = get_json_string(line, "content");
        if (role && content && !chat_history_add(history, role, content)) {
            ok = 0;
            free(role);
            free(content);
            break;
        }
        free(role);
        free(content);

        char *input_line = get_json_string(line, "history");
        if (input_line && !chat_history_add_input(history, input_line)) {
            ok = 0;
            free(input_line);
            break;
        }
        free(input_line);
    }
    free(line);
    fclose(fp);
    return ok;
}

static int chat_history_truncate(struct chat_history *history) {
    if (!history || !history->enabled) {
        return 1;
    }
    for (size_t i = 0; i < history->len; i++) {
        free(history->items[i].role);
        free(history->items[i].content);
    }
    history->len = 0;

    return 1;
}

static int append_json_message(struct buffer *buf, const char *role, const char *content, int comma) {
    char *r = json_escape(role);
    char *c = json_escape(content);
    if (!r || !c) {
        free(r);
        free(c);
        return 0;
    }

    const char *content_type = strcmp(role, "assistant") == 0 ? "output_text" : "input_text";
    size_t len = strlen(r) + strlen(c) + strlen(content_type) + 96;
    char *msg = malloc(len);
    if (!msg) {
        free(r);
        free(c);
        return 0;
    }
    snprintf(msg, len,
             "%s{\"type\":\"message\",\"role\":%s,\"content\":[{\"type\":\"%s\",\"text\":%s}]}",
             comma ? "," : "", r, content_type, c);
    int ok = append_bytes(buf, msg, strlen(msg));

    free(msg);
    free(r);
    free(c);
    return ok;
}

static int append_tool_definitions(struct buffer *buf);

static const char *effective_system_prompt(void) {
    return system_prompt_text ? system_prompt_text : SYSTEM_PROMPT;
}

static char *build_payload_extra(const char *model, const char *input1, const struct chat_history *history, const char *extra_items) {
    char *m = json_escape(model);
    char *sys = json_escape(effective_system_prompt());
    if (!m || !sys) {
        free(m);
        free(sys);
        return NULL;
    }

    struct buffer buf = {0};
    int ok = append_bytes(&buf, "{\"model\":", strlen("{\"model\":")) &&
             append_bytes(&buf, m, strlen(m)) &&
             append_bytes(&buf, ",\"input\":[{\"type\":\"message\",\"role\":\"system\",\"content\":[{\"type\":\"input_text\",\"text\":", strlen(",\"input\":[{\"type\":\"message\",\"role\":\"system\",\"content\":[{\"type\":\"input_text\",\"text\":")) &&
             append_bytes(&buf, sys, strlen(sys)) &&
             append_bytes(&buf, "}]}", 3);

    if (ok && history && history->include_context) {
        for (size_t i = 0; i < history->len; i++) {
            ok = append_json_message(&buf, history->items[i].role, history->items[i].content, 1);
            if (!ok) {
                break;
            }
        }
    }

    if (ok) {
        ok = append_json_message(&buf, "user", input1, 1);
    }
    if (ok) {
        if (extra_items && *extra_items) {
            ok = append_bytes(&buf, extra_items, strlen(extra_items));
        }
    }
    if (ok) {
        ok = append_bytes(&buf, "]", 1) &&
             append_tool_definitions(&buf) &&
             append_bytes(&buf, ",\"stream\":true,\"store\":false}", strlen(",\"stream\":true,\"store\":false}"));
    }

    free(m);
    free(sys);
    if (!ok) {
        free(buf.data);
        return NULL;
    }
    return buf.data;
}

static int host_has_port(const char *server) {
    const char *path = strchr(server, '/');
    size_t host_len = path ? (size_t)(path - server) : strlen(server);
    const char *colon = memchr(server, ':', host_len);
    return colon != NULL;
}

static char *build_url(const char *server, const char *port) {
    const char *prefix = strstr(server, "://") ? "" : "http://";
    const char *suffix = "";
    if (!strstr(server, "/v1/responses")) {
        suffix = "/v1/responses";
    }

    const char *port_sep = "";
    const char *port_value = "";
    if (!strstr(server, "://") && !host_has_port(server) && port && *port) {
        port_sep = ":";
        port_value = port;
    }

    size_t server_len = strlen(server);
    int trim_slash = server_len > 0 && server[server_len - 1] == '/' && suffix[0] == '/';
    size_t len = strlen(prefix) + server_len + strlen(port_sep) + strlen(port_value) + strlen(suffix) + 1;
    char *url = malloc(len);
    if (!url) {
        return NULL;
    }

    if (trim_slash) {
        snprintf(url, len, "%s%.*s%s%s%s", prefix, (int)(server_len - 1), server, port_sep, port_value, suffix);
    } else {
        snprintf(url, len, "%s%s%s%s%s", prefix, server, port_sep, port_value, suffix);
    }
    return url;
}

static const char *skip_ws(const char *p) {
    while (*p && isspace((unsigned char)*p)) {
        p++;
    }
    return p;
}

static char *parse_json_string_at(const char *p) {
    if (*p != '"') {
        return NULL;
    }
    p++;

    size_t cap = strlen(p) + 1;
    char *out = malloc(cap);
    if (!out) {
        return NULL;
    }

    size_t n = 0;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            switch (*p) {
                case '"': out[n++] = '"'; p++; break;
                case '\\': out[n++] = '\\'; p++; break;
                case '/': out[n++] = '/'; p++; break;
                case 'b': out[n++] = '\b'; p++; break;
                case 'f': out[n++] = '\f'; p++; break;
                case 'n': out[n++] = '\n'; p++; break;
                case 'r': out[n++] = '\r'; p++; break;
                case 't': out[n++] = '\t'; p++; break;
                case 'u':
                    /* Preserve non-ASCII escapes instead of partially decoding UTF-16. */
                    if (isxdigit((unsigned char)p[1]) && isxdigit((unsigned char)p[2]) &&
                        isxdigit((unsigned char)p[3]) && isxdigit((unsigned char)p[4])) {
                        out[n++] = '\\';
                        out[n++] = 'u';
                        memcpy(out + n, p + 1, 4);
                        n += 4;
                        p += 5;
                    } else {
                        out[n++] = 'u';
                        p++;
                    }
                    break;
                default:
                    if (*p) {
                        out[n++] = *p++;
                    }
                    break;
            }
        } else {
            out[n++] = *p++;
        }
    }

    out[n] = '\0';
    return out;
}

static int json_text_complete_object(const char *text) {
    int depth = 0;
    int in_string = 0;
    int escaped = 0;
    for (const char *p = text ? text : ""; *p; p++) {
        if (escaped) {
            escaped = 0;
            continue;
        }
        if (in_string) {
            if (*p == '\\') {
                escaped = 1;
            } else if (*p == '"') {
                in_string = 0;
            }
            continue;
        }
        if (*p == '"') {
            in_string = 1;
        } else if (*p == '{' || *p == '[') {
            depth++;
        } else if (*p == '}' || *p == ']') {
            depth--;
            if (depth < 0) {
                return 0;
            }
        }
    }
    return depth == 0 && !in_string && text && (*skip_ws(text) == '{' || *skip_ws(text) == '[');
}

static char *parse_json_value_at(const char *p) {
    p = skip_ws(p);
    if (*p == '"') {
        return parse_json_string_at(p);
    }
    if (*p != '{' && *p != '[') {
        return NULL;
    }

    const char *start = p;
    int depth = 0;
    int in_string = 0;
    int escaped = 0;
    for (; *p; p++) {
        if (escaped) {
            escaped = 0;
            continue;
        }
        if (in_string) {
            if (*p == '\\') {
                escaped = 1;
            } else if (*p == '"') {
                in_string = 0;
            }
            continue;
        }
        if (*p == '"') {
            in_string = 1;
        } else if (*p == '{' || *p == '[') {
            depth++;
        } else if (*p == '}' || *p == ']') {
            depth--;
            if (depth == 0) {
                p++;
                size_t len = (size_t)(p - start);
                char *out = malloc(len + 1);
                if (!out) {
                    return NULL;
                }
                memcpy(out, start, len);
                out[len] = '\0';
                return out;
            }
        }
    }
    return NULL;
}

static const char *find_key(const char *json, const char *key) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);

    const char *p = json;
    while ((p = strstr(p, pattern)) != NULL) {
        const char *q = skip_ws(p + strlen(pattern));
        if (*q == ':') {
            return skip_ws(q + 1);
        }
        p += strlen(pattern);
    }
    return NULL;
}

static char *get_json_string(const char *json, const char *key) {
    const char *p = find_key(json, key);
    if (!p) {
        return NULL;
    }
    return parse_json_string_at(p);
}

static char *get_json_value(const char *json, const char *key) {
    const char *p = find_key(json, key);
    if (!p) {
        return NULL;
    }
    return parse_json_value_at(p);
}

static long get_json_long(const char *json, const char *key) {
    const char *p = find_key(json, key);
    if (!p) {
        return -1;
    }
    return strtol(p, NULL, 10);
}

static int contains_word_ci(const char *s, const char *word) {
    size_t word_len = strlen(word);
    if (word_len == 0) {
        return 1;
    }

    for (; *s; s++) {
        size_t i = 0;
        while (i < word_len && s[i] &&
               tolower((unsigned char)s[i]) == tolower((unsigned char)word[i])) {
            i++;
        }
        if (i == word_len) {
            return 1;
        }
    }
    return 0;
}

static char *get_output_text(const char *json) {
    const char *p = json;
    while ((p = strstr(p, "\"type\"")) != NULL) {
        const char *v = skip_ws(p + 6);
        if (*v++ != ':') {
            p += 6;
            continue;
        }
        v = skip_ws(v);
        char *type = parse_json_string_at(v);
        if (!type) {
            p += 6;
            continue;
        }
        int is_output = strcmp(type, "output_text") == 0;
        free(type);
        if (is_output) {
            const char *t = find_key(p, "text");
            if (t) {
                return parse_json_string_at(t);
            }
        }
        p += 6;
    }

    return get_json_string(json, "output_text");
}

static void update_stream_usage(struct stream_state *st, const char *json) {
    char *model = get_json_string(json, "model");
    if (model) {
        free(st->used_model);
        st->used_model = model;
    }

    long input_tokens = get_json_long(json, "input_tokens");
    long output_tokens = get_json_long(json, "output_tokens");
    long total_tokens = get_json_long(json, "total_tokens");
    long context_used = get_json_long(json, "context_used");

    if (input_tokens < 0) {
        input_tokens = get_json_long(json, "prompt_tokens");
    }
    if (output_tokens < 0) {
        output_tokens = get_json_long(json, "completion_tokens");
    }
    if (total_tokens < 0 && input_tokens >= 0 && output_tokens >= 0) {
        total_tokens = input_tokens + output_tokens;
    } else if (total_tokens < 0) {
        total_tokens = get_json_long(json, "total_tokens");
    }
    if (context_used < 0) {
        context_used = get_json_long(json, "context_used_tokens");
    }
    if (context_used < 0) {
        context_used = get_json_long(json, "prompt_tokens");
    }
    if (context_used < 0) {
        context_used = input_tokens;
    }

    if (input_tokens >= 0) {
        st->input_tokens = input_tokens;
    }
    if (output_tokens >= 0) {
        st->output_tokens = output_tokens;
    }
    if (total_tokens >= 0) {
        st->total_tokens = total_tokens;
    }
    if (context_used >= 0) {
        st->context_used_tokens = context_used;
    }
}

static void reset_stream_indicator_color(struct stream_state *st) {
    if (st->answer_color) {
        fputs(Q_COLOR_RESET, stderr);
    }
}

static void show_stream_indicator(struct stream_state *st, int *active, const char **frames, int frame_count, int thinking) {
    if (isatty(STDERR_FILENO)) {
        reset_stream_indicator_color(st);
        if (thinking && st->answer_started && !*active) {
            fputc('\n', stderr);
        }
        if (st->quiet == 2) {
            fprintf(stderr, "\rcompletion %s", frames[st->thinking_frame % frame_count]);
        } else if (thinking) {
            fprintf(stderr, "\r[LLM Thinking - TURN %02d] %s",
                    st->turn_number > 0 ? st->turn_number : 1,
                    frames[st->thinking_frame % frame_count]);
        } else {
            fprintf(stderr, "\r[LLM] %s", frames[st->thinking_frame % frame_count]);
        }
    } else if (!*active) {
        reset_stream_indicator_color(st);
        fputs(st->quiet == 2 ? "completion ..." : "[LLM] ...", stderr);
    }
    fflush(stderr);
    *active = 1;
    st->thinking_frame++;
}

static void clear_stream_indicator(int *active) {
    if (!*active) {
        return;
    }

    if (isatty(STDERR_FILENO)) {
        fputs("\r\033[K", stderr);
    } else {
        fputc('\n', stderr);
    }
    fflush(stderr);
    *active = 0;
}

static void show_thinking_indicator(struct stream_state *st) {
    const char *frames[] = {"[    ] ","[=   ] ", "[==  ] ", "[=== ] ", "[====] ", "[ ===] ", "[  ==] ", "[   =] " };
    int n = sizeof(frames) / sizeof(frames[0]);

    if (st->think_loud || st->quiet == 1) {
        return;
    }

    show_stream_indicator(st, &st->thinking_indicator, frames, n, 1);
}

static void clear_thinking_indicator(struct stream_state *st) {
    clear_stream_indicator(&st->thinking_indicator);
}

static void show_waiting_indicator(struct stream_state *st) {
    const char *frames[] = {".  ", ".. ", "..."};

    if (st->received_first_byte || st->quiet == 1) {
        return;
    }

    show_stream_indicator(st, &st->waiting_indicator, frames, 3, 0);
}

static void clear_waiting_indicator(struct stream_state *st) {
    clear_stream_indicator(&st->waiting_indicator);
}

static void finish_llm_indicator_line(struct stream_state *st) {
    if (!st) {
        return;
    }
    if (st->waiting_indicator || st->thinking_indicator) {
        if (st->answer_color) {
            fputs(Q_COLOR_RESET, stderr);
        }
        fputc('\n', stderr);
        fflush(stderr);
    }
    st->waiting_indicator = 0;
    st->thinking_indicator = 0;
    st->answer_started = 1;
}

static void preserve_llm_indicator_line(struct stream_state *st) {
    if (!st) {
        return;
    }
    if (st->waiting_indicator || st->thinking_indicator) {
        if (st->answer_color) {
            fputs(Q_COLOR_RESET, stderr);
        }
        fputc('\n', stderr);
        fflush(stderr);
    }
    st->waiting_indicator = 0;
    st->thinking_indicator = 0;
}

static void poll_request_key(struct stream_state *st) {
    if (!st || st->quiet || !isatty(STDIN_FILENO)) {
        return;
    }

    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(STDIN_FILENO, &rfds);
    struct timeval tv = {0, 0};
    int ready = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
    if (ready <= 0 || !FD_ISSET(STDIN_FILENO, &rfds)) {
        return;
    }

    unsigned char c = 0;
    if (read(STDIN_FILENO, &c, 1) != 1) {
        return;
    }
    if (c == 't' || c == 'T') {
        st->think_loud = !st->think_loud;
        clear_waiting_indicator(st);
        clear_thinking_indicator(st);
        fprintf(stderr, "\n/think-loud: changing to %s via t\n", st->think_loud ? "on" : "off");
        fflush(stderr);
    }
}

static int curl_progress_cb(void *clientp, curl_off_t dltotal, curl_off_t dlnow,
                            curl_off_t ultotal, curl_off_t ulnow) {
    struct stream_state *st = clientp;
    (void)dltotal;
    (void)dlnow;
    (void)ultotal;
    (void)ulnow;
    if (request_interrupted) {
        return 1;
    }
    poll_request_key(st);
    show_waiting_indicator(st);
    return 0;
}

static int begin_request_key_mode(struct termios *orig) {
    if (!isatty(STDIN_FILENO) || terminal_raw_active) {
        return 0;
    }
    if (tcgetattr(STDIN_FILENO, orig) != 0) {
        return 0;
    }
    struct termios t = *orig;
    t.c_lflag &= (tcflag_t)~(ECHO | ICANON);
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &t) != 0) {
        return 0;
    }
    return 1;
}

static void end_request_key_mode(const struct termios *orig, int active) {
    if (active) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, orig);
    }
}

static int running_on_linux_virtual_tty(void) {
    const char *term = getenv("TERM");
    return term && strcmp(term, "linux") == 0;
}

static const char *signal_name_for_message(int sig) {
    switch (sig) {
        case SIGINT: return "SIGINT";
        case SIGHUP: return "SIGHUP";
        case SIGTERM: return "SIGTERM";
        case SIGQUIT: return "SIGQUIT";
        default: return "signal";
    }
}

static const char *signal_keys_for_message(int sig) {
    switch (sig) {
        case SIGINT: return "C-c";
        case SIGQUIT: return "C-\\";
        default: return "n/a";
    }
}

static void set_answer_color(struct stream_state *st, const char *code) {
    if (!st->answer_color) {
        return;
    }
    int code_id = 2;
    if (strcmp(code, Q_COLOR_CODE_EMULATOR) == 0) {
        code_id = 1;
    } else if (strcmp(code, Q_COLOR_CODE_TTY) == 0) {
        code_id = 3;
    }
    if (!st->answer_color_started || st->answer_color_code != code_id) {
        if (st->answer_color_started) {
            fputs(Q_COLOR_RESET, stdout);
        }
        fputs(code, stdout);
        fflush(stdout);
        st->answer_color_started = 1;
        st->answer_color_code = code_id;
    }
}

static void reset_answer_color(struct stream_state *st) {
    if (st->answer_color_started) {
        fputs(Q_COLOR_RESET, stdout);
        fflush(stdout);
        st->answer_color_started = 0;
    }
}

static void write_answer_char(struct stream_state *st, char c) {
    set_answer_color(st, st->in_code_fence ? st->code_color : Q_COLOR_LLM_TEXT);
    fputc(c, stdout);
}

static void flush_pending_backticks(struct stream_state *st) {
    while (st->pending_backticks > 0) {
        write_answer_char(st, '`');
        st->pending_backticks--;
    }
}

static void write_answer_text(struct stream_state *st, const char *text) {
    for (const char *p = text; *p; p++) {
        if (*p == '`') {
            st->pending_backticks++;
            if (st->pending_backticks == 3) {
                int opening = !st->in_code_fence;
                if (opening) {
                    st->in_code_fence = 1;
                }
                write_answer_char(st, '`');
                write_answer_char(st, '`');
                write_answer_char(st, '`');
                if (!opening) {
                    st->in_code_fence = 0;
                }
                st->pending_backticks = 0;
            }
            continue;
        }

        flush_pending_backticks(st);
        write_answer_char(st, *p);
    }
    fflush(stdout);
}

static void print_stream_text(struct stream_state *st, char *text) {
    if (!text) {
        return;
    }
    finish_llm_indicator_line(st);
    append_bytes(&st->answer_text, text, strlen(text));
    if (!st->quiet) {
        write_answer_text(st, text);
    }
    st->printed = 1;
    free(text);
}

static void print_thinking_text(struct stream_state *st, char *text) {
    if (!text) {
        return;
    }
    finish_llm_indicator_line(st);
    if (!st->quiet) {
        int saved_in_code = st->in_code_fence;
        int saved_pending = st->pending_backticks;
        st->in_code_fence = 0;
        st->pending_backticks = 0;
        write_answer_text(st, text);
        flush_pending_backticks(st);
        reset_answer_color(st);
        st->in_code_fence = saved_in_code;
        st->pending_backticks = saved_pending;
    }
    free(text);
}

static char *get_first_json_string(const char *json, const char **keys, size_t key_count) {
    for (size_t i = 0; i < key_count; i++) {
        char *value = get_json_string(json, keys[i]);
        if (value) {
            return value;
        }
    }
    return NULL;
}

static int json_has_key(const char *json, const char *key) {
    return find_key(json, key) != NULL;
}

static int replace_or_append_tool_arguments(struct stream_state *st, const char *arguments, const char *type) {
    if (!arguments || !*arguments) {
        return 1;
    }
    int is_delta = (type && contains_word_ci(type, "delta")) || !json_text_complete_object(arguments);
    if (!st->tool_arguments) {
        st->tool_arguments = strdup(arguments);
        return st->tool_arguments != NULL;
    }
    if (json_text_complete_object(st->tool_arguments) && !is_delta) {
        free(st->tool_arguments);
        st->tool_arguments = strdup(arguments);
        return st->tool_arguments != NULL;
    }
    if (!json_text_complete_object(st->tool_arguments)) {
        size_t old_len = strlen(st->tool_arguments);
        size_t add_len = strlen(arguments);
        char *next = realloc(st->tool_arguments, old_len + add_len + 1);
        if (!next) {
            return 0;
        }
        st->tool_arguments = next;
        memcpy(st->tool_arguments + old_len, arguments, add_len + 1);
    }
    return 1;
}

static int is_thinking_json(const char *json, const char *type) {
    if (type && (contains_word_ci(type, "reasoning") ||
                 contains_word_ci(type, "thinking") ||
                 contains_word_ci(type, "thought"))) {
        return 1;
    }

    return json_has_key(json, "reasoning_content") ||
           json_has_key(json, "reasoning") ||
           json_has_key(json, "thinking") ||
           json_has_key(json, "thought");
}

static char *get_thinking_text(const char *json) {
    const char *keys[] = {
        "reasoning_content",
        "reasoning",
        "thinking",
        "thought",
        "delta",
        "text",
        "content"
    };

    return get_first_json_string(json, keys, sizeof(keys) / sizeof(keys[0]));
}

static int note_tool_call(struct stream_state *st, const char *json, const char *type) {
    int looks_like_tool =
        (type && (contains_word_ci(type, "tool_call") ||
                  contains_word_ci(type, "function_call") ||
                  contains_word_ci(type, "tool_calls"))) ||
        json_has_key(json, "tool_calls") ||
        json_has_key(json, "tool_call") ||
        json_has_key(json, "function_call") ||
        contains_word_ci(json, "\"function_call\"") ||
        contains_word_ci(json, "\"tool_call\"");
    if (!looks_like_tool) {
        return 0;
    }

    char *name = get_json_string(json, "name");
    char *arguments = get_json_string(json, "arguments");
    if (!arguments) {
        arguments = get_json_value(json, "arguments");
    }
    if (!arguments && type && contains_word_ci(type, "function_call_arguments")) {
        arguments = get_json_string(json, "delta");
    }
    if (!arguments) {
        arguments = get_json_string(json, "arguments_delta");
    }
    char *input = get_json_string(json, "input");
    if (!input) {
        input = get_json_value(json, "input");
    }
    char *call_id = get_json_string(json, "call_id");
    if (!call_id) {
        call_id = get_json_string(json, "id");
    }

    if (!st->quiet) {
        if (st->tool_calls.len > 0) {
            append_bytes(&st->tool_calls, "\n", 1);
        }
        append_bytes(&st->tool_calls, "Tool call", strlen("Tool call"));
        if (type) {
            append_bytes(&st->tool_calls, " type=", strlen(" type="));
            append_bytes(&st->tool_calls, type, strlen(type));
        }
    }
    if (name) {
        if (!st->quiet) {
            append_bytes(&st->tool_calls, " name=", strlen(" name="));
            append_bytes(&st->tool_calls, name, strlen(name));
        }
        if (!st->tool_name) {
            st->tool_name = strdup(name);
        }
    }
    if (arguments) {
        if (!st->quiet) {
            append_bytes(&st->tool_calls, " arguments=", strlen(" arguments="));
            append_bytes(&st->tool_calls, arguments, strlen(arguments));
        }
        replace_or_append_tool_arguments(st, arguments, type);
    } else if (input) {
        if (!st->quiet) {
            append_bytes(&st->tool_calls, " input=", strlen(" input="));
            append_bytes(&st->tool_calls, input, strlen(input));
        }
        replace_or_append_tool_arguments(st, input, type);
    } else if (!st->quiet) {
        append_bytes(&st->tool_calls, " json=", strlen(" json="));
        append_bytes(&st->tool_calls, json, strlen(json));
    }
    if (call_id && !st->tool_call_id) {
        st->tool_call_id = strdup(call_id);
    }

    free(name);
    free(arguments);
    free(input);
    free(call_id);
    return 1;
}

static void process_sse_json(struct stream_state *st, const char *json) {
    update_stream_usage(st, json);

    char *type = get_json_string(json, "type");
    if (note_tool_call(st, json, type)) {
        free(type);
        return;
    }
    if (is_thinking_json(json, type)) {
        char *thinking = get_thinking_text(json);
        if (thinking && *thinking) {
            if (st->think_loud) {
                print_thinking_text(st, thinking);
            } else {
                show_thinking_indicator(st);
                free(thinking);
            }
        } else {
            free(thinking);
        }
        free(type);
        return;
    }

    if (type && strcmp(type, "response.output_text.delta") == 0) {
        print_stream_text(st, get_json_string(json, "delta"));
        free(type);
        return;
    }
    free(type);

    char *delta = get_json_string(json, "delta");
    if (delta) {
        print_stream_text(st, delta);
        return;
    }

    char *content = get_json_string(json, "content");
    if (content) {
        print_stream_text(st, content);
    }
}

static void process_sse_line(struct stream_state *st, const char *line) {
    line = skip_ws(line);
    if (strncmp(line, "data:", 5) != 0) {
        return;
    }

    const char *data = skip_ws(line + 5);
    if (strcmp(data, "[DONE]") == 0) {
        return;
    }
    if (*data == '{') {
        process_sse_json(st, data);
    }
}

static size_t stream_write_cb(char *ptr, size_t size, size_t nmemb, void *userdata) {
    size_t n = size * nmemb;
    struct stream_state *st = userdata;

    if (request_interrupted) {
        return 0;
    }

    poll_request_key(st);

    if (!st->received_first_byte) {
        st->received_first_byte = 1;
    }

    if (!append_bytes(&st->raw, ptr, n)) {
        return 0;
    }

    size_t start = 0;
    for (size_t i = 0; i < n; i++) {
        if (ptr[i] != '\n') {
            continue;
        }

        if (!append_bytes(&st->line, ptr + start, i - start)) {
            return 0;
        }
        if (st->line.len > 0 && st->line.data[st->line.len - 1] == '\r') {
            st->line.data[--st->line.len] = '\0';
        }
        process_sse_line(st, st->line.data ? st->line.data : "");
        st->line.len = 0;
        if (st->line.data) {
            st->line.data[0] = '\0';
        }
        start = i + 1;
    }

    if (start < n && !append_bytes(&st->line, ptr + start, n - start)) {
        return 0;
    }

    return n;
}

static void flush_stream_line(struct stream_state *st) {
    if (!st || !st->line.data || st->line.len == 0) {
        return;
    }
    if (st->line.data[st->line.len - 1] == '\r') {
        st->line.data[--st->line.len] = '\0';
    }
    process_sse_line(st, st->line.data);
    st->line.len = 0;
    st->line.data[0] = '\0';
}

static void print_api_error_suggestion(long status, const char *message, const char *response) {
    const char *text = message && *message ? message : (response ? response : "");

    if (contains_word_ci(text, "context") ||
        contains_word_ci(text, "maximum context") ||
        contains_word_ci(text, "context length") ||
        contains_word_ci(text, "token limit") ||
        contains_word_ci(text, "too many tokens")) {
        fprintf(stderr, "Suggestion: reduce input, lower context, or run /truncate-context in REPL.\n");
        return;
    }
    if (contains_word_ci(text, "rate limit") ||
        contains_word_ci(text, "too many requests") ||
        status == 429) {
        fprintf(stderr, "Suggestion: wait and retry, lower request rate, reduce parallel clients, or check the server rate-limit configuration.\n");
        return;
    }
    if (contains_word_ci(text, "model") &&
        (contains_word_ci(text, "not found") || contains_word_ci(text, "unknown") || contains_word_ci(text, "not loaded"))) {
        fprintf(stderr, "Suggestion: check OPENAI_MODEL, available local models, and whether the model is loaded on the server.\n");
        return;
    }
    if (contains_word_ci(text, "tool") ||
        contains_word_ci(text, "function") ||
        contains_word_ci(text, "schema")) {
        fprintf(stderr, "Suggestion: check tool/function-call compatibility with this server and inspect --api-logging query output.\n");
        return;
    }

    switch (status) {
        case 400:
            fprintf(stderr, "Suggestion: request was rejected as invalid; inspect --api-logging query for payload/schema incompatibility.\n");
            break;
        case 401:
            fprintf(stderr, "Suggestion: authentication failed; check OPENAI_API_KEY or leave it empty for local servers that do not require it.\n");
            break;
        case 403:
            fprintf(stderr, "Suggestion: access was forbidden; check API key permissions, model permissions, or server policy.\n");
            break;
        case 404:
            fprintf(stderr, "Suggestion: endpoint or model was not found; check LLM_SERVER, LLM_PORT, /v1/responses support, and OPENAI_MODEL.\n");
            break;
        case 408:
            fprintf(stderr, "Suggestion: request timed out server-side; retry, reduce input, or increase server/model timeout settings.\n");
            break;
        case 413:
            fprintf(stderr, "Suggestion: request payload is too large; reduce prompt/context size or disable --keep-context.\n");
            break;
        case 422:
            fprintf(stderr, "Suggestion: server could not process the payload; check model name, tool schema support, and OpenAI API format compatibility.\n");
            break;
        case 500:
            fprintf(stderr, "Suggestion: server hit an internal error; check LLM server logs and model backend health.\n");
            break;
        case 502:
        case 503:
        case 504:
            fprintf(stderr, "Suggestion: server/backend is unavailable or overloaded; check model process, GPU/CPU memory, proxy, and retry.\n");
            break;
        default:
            if (status >= 400 && status < 500) {
                fprintf(stderr, "Suggestion: client/request error; inspect --api-logging query and server API compatibility.\n");
            } else if (status >= 500) {
                fprintf(stderr, "Suggestion: server/backend error; check local LLM logs, model availability, and resource usage.\n");
            } else {
                fprintf(stderr, "Suggestion: inspect --api-logging both and the LLM server logs.\n");
            }
            break;
    }
}

static void print_error_response(long status, const char *response) {
    char *message = get_json_string(response, "message");
    char *error_type = get_json_string(response, "type");
    char *error_code = get_json_string(response, "code");
    fprintf(stderr, "OpenAI API error");
    if (status > 0) {
        fprintf(stderr, " (HTTP %ld)", status);
    }
    fprintf(stderr, "\n");
    if (error_type || error_code) {
        fprintf(stderr, "Type: %s%s%s\n",
                error_type ? error_type : "",
                error_type && error_code ? ", code: " : "",
                error_code ? error_code : "");
    }
    if (message) {
        fprintf(stderr, "Message: %s\n", message);
    } else {
        fprintf(stderr, "Response:\n%s\n", response ? response : "");
    }
    print_api_error_suggestion(status, message, response);
    free(message);
    free(error_type);
    free(error_code);
}

static int confirm_file_tool_operation(const char *name, const char *path, int color) {
    if (!isatty(STDIN_FILENO)) {
        fprintf(stderr, "%s %s? confirmation unavailable without a TTY\n", name, path);
        return 0;
    }

    if (color && isatty(STDERR_FILENO)) {
        fprintf(stderr, Q_COLOR_TOOL_CONFIRM "[Tool Calling] %s %s? [y/N] " Q_COLOR_RESET, name, path);
    } else {
        fprintf(stderr, "[Tool Calling] %s %s? [y/N] ", name, path);
    }
    fflush(stderr);

    struct termios orig;
    int raw = 0;
    if (tcgetattr(STDIN_FILENO, &orig) == 0) {
        struct termios t = orig;
        t.c_lflag &= (tcflag_t)~(ICANON | ECHO);
        t.c_cc[VMIN] = 1;
        t.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &t) == 0) {
            raw = 1;
        }
    }

    unsigned char c = 0;
    int ok = read(STDIN_FILENO, &c, 1) == 1 && (c == 'y' || c == 'Y');
    if (raw) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig);
    }
    fprintf(stderr, "%c\n", c ? c : 'n');
    return ok;
}

static int prompt_overwrite_or_alternate(const char *path, int color) {
    if (!isatty(STDIN_FILENO)) {
        fprintf(stderr, "[Tool Calling] write_file %s exists; confirmation unavailable without a TTY\n", path);
        return 'n';
    }
    if (color && isatty(STDERR_FILENO)) {
        fprintf(stderr, Q_COLOR_TOOL_CONFIRM "[Tool Calling] write_file %s exists. overwrite/alternate/no? [o/a/N] " Q_COLOR_RESET, path);
    } else {
        fprintf(stderr, "[Tool Calling] write_file %s exists. overwrite/alternate/no? [o/a/N] ", path);
    }
    fflush(stderr);

    struct termios orig;
    int raw = 0;
    if (tcgetattr(STDIN_FILENO, &orig) == 0) {
        struct termios t = orig;
        t.c_lflag &= (tcflag_t)~(ICANON | ECHO);
        t.c_cc[VMIN] = 1;
        t.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &t) == 0) {
            raw = 1;
        }
    }

    unsigned char c = 0;
    if (read(STDIN_FILENO, &c, 1) != 1) {
        c = 'n';
    }
    if (raw) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig);
    }
    fprintf(stderr, "%c\n", c ? c : 'n');
    if (c == 'o' || c == 'O' || c == 'y' || c == 'Y') {
        return 'o';
    }
    if (c == 'a' || c == 'A') {
        return 'a';
    }
    return 'n';
}

static char *prompt_alternate_path(int color) {
    if (!isatty(STDIN_FILENO)) {
        fprintf(stderr, "[Tool Calling] alternate path unavailable without a TTY\n");
        return NULL;
    }
    if (color && isatty(STDERR_FILENO)) {
        fprintf(stderr, Q_COLOR_TOOL_CONFIRM "[Tool Calling] alternate path: " Q_COLOR_RESET);
    } else {
        fprintf(stderr, "[Tool Calling] alternate path: ");
    }
    fflush(stderr);

    char *line = NULL;
    size_t cap = 0;
    ssize_t n = getline(&line, &cap, stdin);
    if (n < 0) {
        free(line);
        return NULL;
    }
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
        line[--n] = '\0';
    }
    char *trimmed = (char *)skip_ws(line);
    if (!*trimmed) {
        free(line);
        return NULL;
    }
    char *out = strdup(trimmed);
    free(line);
    return out;
}

static void print_tool_permission_error(const char *operation, const char *path, int color) {
    if (color && isatty(STDERR_FILENO)) {
        fprintf(stderr, Q_COLOR_EXIT_FAIL "[Tool Calling] %s %s: permission denied" Q_COLOR_RESET "\n",
                operation, path);
    } else {
        fprintf(stderr, "[Tool Calling] %s %s: permission denied\n", operation, path);
    }
}

static char *parent_dir_for_path(const char *path) {
    const char *slash = strrchr(path, '/');
    if (!slash) {
        return strdup(".");
    }
    if (slash == path) {
        return strdup("/");
    }
    size_t len = (size_t)(slash - path);
    char *dir = malloc(len + 1);
    if (!dir) {
        return NULL;
    }
    memcpy(dir, path, len);
    dir[len] = '\0';
    return dir;
}

static int can_write_target_path(const char *path, int color) {
    struct stat st;
    if (stat(path, &st) == 0) {
        if (S_ISDIR(st.st_mode)) {
            if (color && isatty(STDERR_FILENO)) {
                fprintf(stderr, Q_COLOR_EXIT_FAIL "[Tool Calling] write_file %s: target is a directory" Q_COLOR_RESET "\n", path);
            } else {
                fprintf(stderr, "[Tool Calling] write_file %s: target is a directory\n", path);
            }
            return 0;
        }
        if (access(path, W_OK) != 0) {
            if (errno == EACCES || errno == EPERM) {
                print_tool_permission_error("write_file", path, color);
            } else {
                perror("write_file access");
            }
            return 0;
        }
        return 1;
    }

    char *dir = parent_dir_for_path(path);
    if (!dir) {
        fprintf(stderr, "[Tool Calling] write_file %s: memory allocation failed\n", path);
        return 0;
    }
    int ok = access(dir, W_OK | X_OK) == 0;
    if (!ok) {
        if (errno == EACCES || errno == EPERM) {
            print_tool_permission_error("write_file", dir, color);
        } else {
            fprintf(stderr, "[Tool Calling] write_file %s: cannot write in parent directory %s: %s\n",
                    path, dir, strerror(errno));
        }
    }
    free(dir);
    return ok;
}

static char *resolve_write_target_path(const char *initial_path, int color, int *declined) {
    char *path = strdup(initial_path);
    if (!path) {
        return NULL;
    }
    *declined = 0;

    while (1) {
        if (!can_write_target_path(path, color)) {
            free(path);
            return NULL;
        }

        struct stat st;
        if (stat(path, &st) != 0) {
            return path;
        }

        int choice = prompt_overwrite_or_alternate(path, color);
        if (choice == 'o') {
            return path;
        }
        if (choice != 'a') {
            *declined = 1;
            free(path);
            return NULL;
        }

        char *alternate = prompt_alternate_path(color);
        if (!alternate) {
            *declined = 1;
            free(path);
            return NULL;
        }
        free(path);
        path = alternate;
    }
}

static char *execute_local_tool_call(const char *name, const char *arguments, int color) {
    if (!name) {
        return strdup("{\"ok\":false,\"error\":\"missing tool name\"}");
    }
    if (!arguments) {
        arguments = "{}";
    }

    char *mcp_output = execute_mcp_tool_call(name, arguments);
    if (mcp_output) {
        return mcp_output;
    }

    if (strcmp(name, "get_time") == 0) {
        time_t now = time(NULL);
        struct tm local_tm;
        struct tm utc_tm;
        localtime_r(&now, &local_tm);
        gmtime_r(&now, &utc_tm);

        char local_time[64];
        char utc_time[64];
        char timezone_name[32];
        char timezone_offset[16];
        strftime(local_time, sizeof(local_time), "%Y-%m-%dT%H:%M:%S%z", &local_tm);
        strftime(utc_time, sizeof(utc_time), "%Y-%m-%dT%H:%M:%SZ", &utc_tm);
        strftime(timezone_name, sizeof(timezone_name), "%Z", &local_tm);
        strftime(timezone_offset, sizeof(timezone_offset), "%z", &local_tm);

        char *local_json = json_escape(local_time);
        char *utc_json = json_escape(utc_time);
        char *tz_json = json_escape(timezone_name);
        char *offset_json = json_escape(timezone_offset);
        if (!local_json || !utc_json || !tz_json || !offset_json) {
            free(local_json);
            free(utc_json);
            free(tz_json);
            free(offset_json);
            return strdup("{\"ok\":false,\"error\":\"memory allocation failed\"}");
        }

        size_t len = strlen(local_json) + strlen(utc_json) + strlen(tz_json) + strlen(offset_json) + 128;
        char *out = malloc(len);
        if (!out) {
            free(local_json);
            free(utc_json);
            free(tz_json);
            free(offset_json);
            return strdup("{\"ok\":false,\"error\":\"memory allocation failed\"}");
        }
        snprintf(out, len,
                 "{\"ok\":true,\"unix\":%lld,\"local\":%s,\"utc\":%s,\"timezone\":%s,\"utc_offset\":%s}",
                 (long long)now, local_json, utc_json, tz_json, offset_json);
        free(local_json);
        free(utc_json);
        free(tz_json);
        free(offset_json);
        return out;
    }

    char *path = get_json_string(arguments, "path");
    if (!path || !*path) {
        free(path);
        return strdup("{\"ok\":false,\"error\":\"missing path\"}");
    }

    if (strcmp(name, "read_file") == 0) {
        if (access(path, F_OK) != 0) {
            char *escaped_path = json_escape(path);
            free(path);
            if (!escaped_path) {
                return strdup("{\"ok\":false,\"error\":\"memory allocation failed\"}");
            }
            size_t len = strlen(escaped_path) + 96;
            char *out = malloc(len);
            if (!out) {
                free(escaped_path);
                return strdup("{\"ok\":false,\"error\":\"memory allocation failed\"}");
            }
            snprintf(out, len, "{\"ok\":false,\"path\":%s,\"error\":\"file is not present\"}", escaped_path);
            free(escaped_path);
            return out;
        }
        if (access(path, R_OK) != 0) {
            if (errno == EACCES || errno == EPERM) {
                print_tool_permission_error("read_file", path, color);
            } else {
                perror("read_file access");
            }
            free(path);
            return NULL;
        }
        if (!confirm_file_tool_operation("read_file", path, color)) {
            free(path);
            return strdup("{\"ok\":false,\"error\":\"user declined read_file\"}");
        }
        char *content = read_text_file(path);
        char *escaped_path = json_escape(path);
        char *escaped_content = json_escape(content ? content : "");
        free(path);
        free(content);
        if (!escaped_path || !escaped_content) {
            free(escaped_path);
            free(escaped_content);
            return strdup("{\"ok\":false,\"error\":\"memory allocation failed\"}");
        }
        size_t len = strlen(escaped_path) + strlen(escaped_content) + 64;
        char *out = malloc(len);
        if (!out) {
            free(escaped_path);
            free(escaped_content);
            return strdup("{\"ok\":false,\"error\":\"memory allocation failed\"}");
        }
        snprintf(out, len, "{\"ok\":true,\"path\":%s,\"content\":%s}", escaped_path, escaped_content);
        free(escaped_path);
        free(escaped_content);
        return out;
    }

    if (strcmp(name, "write_file") == 0) {
        char *content = get_json_string(arguments, "content");
        if (!content) {
            free(path);
            return strdup("{\"ok\":false,\"error\":\"missing content\"}");
        }
        int declined = 0;
        char *target_path = resolve_write_target_path(path, color, &declined);
        free(path);
        if (!target_path) {
            free(content);
            if (declined) {
                return strdup("{\"ok\":false,\"error\":\"user declined write_file\"}");
            }
            return NULL;
        }
        if (!confirm_file_tool_operation("write_file", target_path, color)) {
            free(target_path);
            free(content);
            return strdup("{\"ok\":false,\"error\":\"user declined write_file\"}");
        }
        int ok = write_text_file(target_path, content);
        char *escaped_path = json_escape(target_path);
        free(target_path);
        free(content);
        if (!escaped_path) {
            return strdup("{\"ok\":false,\"error\":\"memory allocation failed\"}");
        }
        size_t len = strlen(escaped_path) + 80;
        char *out = malloc(len);
        if (!out) {
            free(escaped_path);
            return strdup("{\"ok\":false,\"error\":\"memory allocation failed\"}");
        }
        snprintf(out, len, "{\"ok\":%s,\"path\":%s}", ok ? "true" : "false", escaped_path);
        free(escaped_path);
        return out;
    }

    free(path);
    return strdup("{\"ok\":false,\"error\":\"unknown tool\"}");
}

static char *infer_tool_name_from_arguments(const char *arguments) {
    if (!arguments) {
        return NULL;
    }
    char *path = get_json_string(arguments, "path");
    if (!path) {
        return NULL;
    }
    char *content = get_json_string(arguments, "content");
    free(path);
    if (content) {
        free(content);
        return strdup("write_file");
    }
    return strdup("read_file");
}

static char *build_tool_result_items(const char *call_id, const char *name, const char *arguments, const char *output) {
    char *cid = json_escape(call_id ? call_id : "call_0");
    char *n = json_escape(name ? name : "");
    char *args = json_escape(arguments ? arguments : "{}");
    char *out = json_escape(output ? output : "");
    if (!cid || !n || !args || !out) {
        free(cid);
        free(n);
        free(args);
        free(out);
        return NULL;
    }

    const char *fmt =
        ",{\"type\":\"function_call\",\"call_id\":%s,\"name\":%s,\"arguments\":%s}"
        ",{\"type\":\"function_call_output\",\"call_id\":%s,\"output\":%s}";
    size_t len = snprintf(NULL, 0, fmt, cid, n, args, cid, out);
    char *items = malloc(len + 1);
    if (items) {
        snprintf(items, len + 1, fmt, cid, n, args, cid, out);
    }
    free(cid);
    free(n);
    free(args);
    free(out);
    return items;
}

static void print_key_bindings(FILE *out) {
    fprintf(out,
            "\nKey bindings:\n"
            "  C-p                  Previous history item.\n"
            "  C-n                  Next history item.\n"
            "  C-a / Home           Beginning of line.\n"
            "  C-e / End            End of line.\n"
            "  C-b / Left           Backward char.\n"
            "  C-f / Right          Forward char.\n"
            "  C-Shift-b            Backward word when supported by the terminal.\n"
            "  C-Shift-f            Forward word when supported by the terminal.\n"
            "  C-w                  Delete previous word.\n"
            "  C-Shift-w            Delete forward word when supported by the terminal.\n"
            "  C-k                  Delete to end of line.\n"
            "  C-d / Delete         Delete char at cursor.\n"
            "  C-l                  Clear screen.\n"
            "  Tab                  Complete current token.\n"
            "  C-c                  Interrupt current LLM request or clear input.\n"
            "  t                    Toggle thinking-loud during an active LLM request.\n");
}

static void print_system_prompt_source(FILE *out) {
    if (system_prompt_path) {
        fprintf(out, "[From file] %s", system_prompt_path);
    } else {
        fprintf(out, "[Builtin]");
    }
}

static void print_masked_api_key(FILE *out) {
    const char *key = getenv("OPENAI_API_KEY");
    if (!key || !*key) {
        fprintf(out, "(empty)");
        return;
    }
    fprintf(out, "%.*s...", 5, key);
}

static void show_system_prompt(FILE *out) {
    fprintf(out, "System prompt source: ");
    print_system_prompt_source(out);
    fprintf(out, "\n%s\n", effective_system_prompt());
}

static void print_repl_help(FILE *out, int show_values, int think_loud, int api_logging) {
    fprintf(out,
            "REPL escapes:\n"
            "  ? <input>            Force input to the LLM.\n"
            "  ! <command>          Force input to the shell.\n"
            "  ??                   Ask the LLM about the previous failing command output.\n"
            "  /help                Show slash commands.\n"
            "  /keys                Show key bindings only.\n"
            "  /show-system-prompt  Show the effective system prompt.");
    if (show_values) {
        fprintf(out, " Current: ");
        print_system_prompt_source(out);
    }
    fprintf(out,
            "\n"
            "  /set-system-prompt existing-filepath\n"
            "                       Use an existing readable file as the system prompt.\n"
            "  /exit                Exit the REPL.\n"
            "  /llm-timeout seconds\n");
    fprintf(out, "                       Change query timeout. Use -1 to wait forever.%s",
            show_values ? " Current: " : "");
    if (show_values) {
        print_llm_timeout_value(out, llm_timeout_seconds);
    }
    fprintf(out,
            "\n"
            "  /llm-turn-limit count\n");
    fprintf(out, "                       Limit tool follow-up turns. Use -1 for unlimited, 0 for none.%s",
            show_values ? " Current: " : "");
    if (show_values) {
        fprintf(out, "%ld", llm_turn_limit);
    }
    fprintf(out,
            "\n"
            "  /think-loud [on|off]\n");
    fprintf(out, "                       Toggle or set reasoning/thinking stream output.%s",
            show_values ? " Current: " : "");
    if (show_values) {
        fprintf(out, "%s", think_loud ? "on" : "off");
    }
    fprintf(out,
            "\n"
            "  /api-logging [none|query|response|both|path]\n");
    fprintf(out, "                       Toggle, set API logging, or append logs to path. Default: none.%s",
            show_values ? " Current: " : "");
    if (show_values) {
        fprintf(out, "%s", api_logging_mode_name(api_logging));
    }
    fprintf(out,
            "\n"
            "  /add-mcp-server url Register an HTTP MCP server by URL.\n"
            "  /remove-mcp-server name\n"
            "                       Remove a registered MCP server.\n"
            "  /list-mcp-servers    List registered MCP servers and alive/tool status.\n"
            "  /note text           Append text to ~/.config/q/NOTES.\n"
            "  /truncate-context    Empty the LLM context without clearing input history.\n"
            "  /clear-completion-cache\n"
            "                       Clear cached command option completions.\n"
            "\n"
            "System prompt: ");
    print_system_prompt_source(out);
    if (show_values) {
        const char *server = getenv("LLM_SERVER");
        const char *port = getenv("LLM_PORT");
        if (!server || !*server) {
            server = DEFAULT_LLM_SERVER;
        }
        if (!port || !*port) {
            port = DEFAULT_LLM_PORT;
        }
        fprintf(out, "\nLLM URL: %s\nLLM port: %s\nAPI key: ", server, port);
        print_masked_api_key(out);
    }
    fprintf(out,
            "\n"
            "Shell aliases from $SHELL are loaded when the REPL starts.\n");
    if (!clipboard_available()) {
        fprintf(out,
                "Warning: copy/paste is disabled! Install one of: wl-clipboard, xclip, xsel.\n");
    }
}

static void print_usage(const char *prog) {
    fprintf(stderr,
            "usage: %s [--repl] [--keep-context] [--record-session] [--resume-session [id|last]] [--list-sessions] [--add-mcp-server url] [--remove-mcp-server name] [--list-mcp-servers] [--llm-timeout seconds] [--llm-turn-limit count] [--api-logging none|query|response|both|path] [--system-prompt-file filepath] [--think-loud] [--color] [query words...]\n"
            "\n"
            "Options:\n"
            "  --repl               Start a unified shell/LLM read-parse-execute loop.\n"
            "  --keep-context       Send prior user/assistant turns from this run on each new request.\n"
            "  --record-session     Save session transcript to ~/.config/q/sessions/session-<timestamp>.\n"
            "  --resume-session [id|last]\n"
            "                       Resume session transcript. Default id: last.\n"
            "  --list-sessions      List recorded sessions and exit.\n"
            "  --add-mcp-server url\n"
            "                       Register an HTTP MCP server URL; name is read from initialize.\n"
            "  --remove-mcp-server name\n"
            "                       Remove a registered MCP server.\n"
            "  --list-mcp-servers   List registered MCP servers and current alive/tool status.\n"
            "  --llm-timeout seconds\n"
            "                       Query timeout in seconds. Use -1 to wait forever. Default: %ld\n"
            "  --llm-turn-limit count\n"
            "                       Limit tool follow-up turns. -1 unlimited, 0 none. Default: %d\n"
            "  --api-logging value  API logging: none, query, response, both, or append silently to path. Default: none\n"
            "  --system-prompt-file filepath\n"
            "                       Use an existing readable file as the system prompt.\n"
            "  --think-loud         Show reasoning/thinking stream output. Default: off\n"
            "  --color              Enable ANSI colors. Default: off\n"
            "\n",
            prog, DEFAULT_LLM_TIMEOUT, DEFAULT_LLM_TURN_LIMIT);
    print_repl_help(stderr, 0, 0, API_LOGGING_NONE);
    fprintf(stderr,
            "\n"
            "Environment:\n"
            "  LLM_SERVER           Local OpenAI-compatible server host or URL. Default: %s\n"
            "  LLM_PORT             Server port when LLM_SERVER has no port. Default: %s\n"
            "  OPENAI_MODEL         Model name sent in the request. Default: %s\n"
            "  OPENAI_API_KEY       Bearer token. Default: empty string\n",
            DEFAULT_LLM_SERVER, DEFAULT_LLM_PORT, DEFAULT_MODEL);
}

static int parse_llm_timeout_value(const char *text, long *out) {
    if (!text || !*text) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    while (end && isspace((unsigned char)*end)) {
        end++;
    }
    if (errno != 0 || !end || *end != '\0' || (value < 1 && value != -1)) {
        return 0;
    }
    *out = value;
    return 1;
}

static void print_llm_timeout_value(FILE *out, long value) {
    fprintf(out, "%ld", value);
    if (value == -1) {
        fprintf(out, " (wait forever)");
    }
}

static int parse_llm_turn_limit_value(const char *text, long *out) {
    if (!text || !*text) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno != 0 || !end || *end != '\0' || value < -1) {
        return 0;
    }
    *out = value;
    return 1;
}

static void print_llm_timeout_help(FILE *out, const char *name) {
    fprintf(out,
            "%s: invalid value\n"
            "Usage: %s seconds\n"
            "Valid values:\n"
            "  -1                  wait forever\n"
            "  positive integer    timeout in seconds\n",
            name, name);
}

static void print_llm_turn_limit_help(FILE *out, const char *name) {
    fprintf(out,
            "%s: invalid value\n"
            "Usage: %s count\n"
            "Valid values:\n"
            "  -1                  unlimited tool follow-up turns\n"
            "  0                   no tool follow-up turns\n"
            "  positive integer    maximum tool follow-up turns after the initial LLM response\n",
            name, name);
}

static void print_think_loud_help(FILE *out, const char *name) {
    fprintf(out,
            "%s: invalid value\n"
            "Usage: %s [on|off]\n"
            "Valid values:\n"
            "  on                  show reasoning/thinking stream output\n"
            "  off                 hide reasoning/thinking stream output\n"
            "  no value            toggle the current value\n",
            name, name);
}

static void print_api_logging_help(FILE *out, const char *name) {
    fprintf(out,
            "%s: invalid value\n"
            "Usage: %s [none|query|response|both|path]\n"
            "Valid values:\n"
            "  none                disable API logging\n"
            "  query               show or log requests sent to the LLM\n"
            "  response            show or log responses received from the LLM\n"
            "  both                show or log requests and responses\n"
            "  writable path       append logs to this file instead of printing them\n"
            "  no value            toggle between none and both (slash command only)\n",
            name, name);
}

static void print_add_mcp_server_help(FILE *out, const char *name) {
    fprintf(out,
            "%s: missing URL\n"
            "Usage: %s http://host:port/mcp\n"
            "Value:\n"
            "  HTTP MCP URL        server name is read from its initialize response\n",
            name, name);
}

static void print_remove_mcp_server_help(FILE *out, const char *name) {
    fprintf(out,
            "%s: missing name\n"
            "Usage: %s name\n"
            "Value:\n"
            "  name                registered MCP server name; use /list-mcp-servers or --list-mcp-servers\n",
            name, name);
}

static void print_system_prompt_file_help(FILE *out, const char *name) {
    fprintf(out,
            "%s: missing file path\n"
            "Usage: %s existing-filepath\n"
            "Value:\n"
            "  existing-filepath   readable regular file to use as the system prompt\n",
            name, name);
}

static void print_resume_session_help(FILE *out, const char *name) {
    fprintf(out,
            "%s: invalid session id\n"
            "Usage: %s [id|last]\n"
            "Valid values:\n"
            "  last                most recent recorded session\n"
            "  positive integer    session id shown by --list-sessions\n"
            "  session-* filename  existing session filename, for compatibility\n"
            "  no value            same as last\n",
            name, name);
}

static void print_slash_commands_brief(FILE *out) {
    fprintf(out,
            "Available slash commands:\n"
            "  /help\n"
            "  /keys\n"
            "  /show-system-prompt\n"
            "  /set-system-prompt existing-filepath\n"
            "  /exit\n"
            "  /llm-timeout seconds\n"
            "  /llm-turn-limit count\n"
            "  /think-loud [on|off]\n"
            "  /api-logging [none|query|response|both|path]\n"
            "  /add-mcp-server url\n"
            "  /remove-mcp-server name\n"
            "  /list-mcp-servers\n"
            "  /note text\n"
            "  /truncate-context\n"
            "  /clear-completion-cache\n");
}

static const char *api_logging_mode_name(int mode) {
    if (api_logging_path) {
        return api_logging_path;
    }
    switch (mode) {
        case API_LOGGING_QUERY: return "query";
        case API_LOGGING_RESPONSE: return "response";
        case API_LOGGING_BOTH: return "both";
        case API_LOGGING_NONE:
        default: return "none";
    }
}

static int parse_api_logging_mode(const char *text, int *out) {
    if (!text || !*text) {
        return 0;
    }
    if (strcmp(text, "none") == 0) {
        *out = API_LOGGING_NONE;
        return 1;
    }
    if (strcmp(text, "query") == 0) {
        *out = API_LOGGING_QUERY;
        return 1;
    }
    if (strcmp(text, "response") == 0) {
        *out = API_LOGGING_RESPONSE;
        return 1;
    }
    if (strcmp(text, "both") == 0) {
        *out = API_LOGGING_BOTH;
        return 1;
    }
    return 0;
}

static int set_api_logging_path(const char *path, int *mode_out) {
    FILE *fp = fopen(path, "a");
    if (!fp) {
        return 0;
    }
    if (fclose(fp) != 0) {
        return 0;
    }
    char *copy = strdup(path);
    if (!copy) {
        return 0;
    }
    free(api_logging_path);
    api_logging_path = copy;
    *mode_out = API_LOGGING_BOTH;
    return 1;
}

static int parse_api_logging_setting(const char *text, int *out) {
    if (parse_api_logging_mode(text, out)) {
        free(api_logging_path);
        api_logging_path = NULL;
        return 1;
    }
    if (text && *text && set_api_logging_path(text, out)) {
        return 1;
    }
    return 0;
}

static FILE *open_api_log_output(int *needs_close) {
    *needs_close = 0;
    if (!api_logging_path) {
        return stderr;
    }
    FILE *fp = fopen(api_logging_path, "a");
    if (!fp) {
        fprintf(stderr, "api logging: cannot append to %s: %s\n", api_logging_path, strerror(errno));
        return NULL;
    }
    *needs_close = 1;
    return fp;
}

static void close_api_log_output(FILE *out, int needs_close) {
    if (needs_close && out) {
        fclose(out);
    }
}

static void log_http_request(const char *url, const char *model, const char *payload, int has_auth) {
    int needs_close = 0;
    FILE *out = open_api_log_output(&needs_close);
    if (!out) {
        return;
    }
    size_t payload_len = strlen(payload);

    fprintf(out, "\nPOST %s HTTP/1.1\n", url);
    fprintf(out, "Content-Type: application/json\n");
    if (has_auth) {
        fprintf(out, "Authorization: Bearer <redacted>\n");
    }
    fprintf(out, "User-Agent: q/1.0\n");
    fprintf(out, "Model: %s\n", model);
    fprintf(out, "Payload-Bytes: %zu\n", payload_len);

    fprintf(out, "\n\n%s\n\n", payload);
    close_api_log_output(out, needs_close);
}

static void log_http_response(long status, const char *response) {
    int needs_close = 0;
    FILE *out = open_api_log_output(&needs_close);
    if (!out) {
        return;
    }
    fprintf(out, "\nHTTP response status: %ld\n", status);
    fprintf(out, "\n\n%s\n\n", response ? response : "");
    close_api_log_output(out, needs_close);
}

static char *first_shell_word(const char *line) {
    line = skip_ws(line);
    if (!*line) {
        return NULL;
    }

    size_t cap = strlen(line) + 1;
    char *word = malloc(cap);
    if (!word) {
        return NULL;
    }

    size_t n = 0;
    char quote = 0;
    for (const char *p = line; *p; p++) {
        if (quote) {
            if (*p == quote) {
                quote = 0;
            } else if (*p == '\\' && p[1]) {
                word[n++] = *++p;
            } else {
                word[n++] = *p;
            }
            continue;
        }

        if (*p == '\'' || *p == '"') {
            quote = *p;
        } else if (*p == '\\' && p[1]) {
            word[n++] = *++p;
        } else if (isspace((unsigned char)*p)) {
            break;
        } else {
            word[n++] = *p;
        }
    }

    word[n] = '\0';
    if (!*word) {
        free(word);
        return NULL;
    }
    return word;
}

static int is_shell_reserved_or_builtin(const char *word) {
    static const char *commands[] = {
        "!", "[", "[[", "{", "}", "alias", "bg", "bind", "break", "builtin",
        "caller", "case", "cd", "command", "compgen", "complete", "compopt",
        "continue", "declare", "dirs", "disown", "do", "done", "echo", "elif",
        "else", "enable", "esac", "eval", "exec", "exit", "export", "false",
        "fc", "fg", "fi", "for", "function", "getopts", "hash", "help",
        "history", "if", "in", "jobs", "kill", "let", "local", "logout",
        "mapfile", "popd", "printf", "pushd", "pwd", "read", "readarray",
        "readonly", "return", "select", "set", "shift", "shopt", "source",
        "suspend", "test", "then", "time", "times", "trap", "true", "type",
        "typeset", "ulimit", "umask", "unalias", "unset", "until", "wait",
        "while"
    };

    for (size_t i = 0; i < sizeof(commands) / sizeof(commands[0]); i++) {
        if (strcmp(word, commands[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

static int is_executable_word(const char *word) {
    if (strchr(word, '/')) {
        return access(word, X_OK) == 0;
    }

    const char *path = getenv("PATH");
    if (!path) {
        path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
    }

    char *copy = strdup(path);
    if (!copy) {
        return 0;
    }

    int found = 0;
    char *save = NULL;
    for (char *dir = strtok_r(copy, ":", &save); dir; dir = strtok_r(NULL, ":", &save)) {
        if (!*dir) {
            dir = ".";
        }
        size_t len = strlen(dir) + 1 + strlen(word) + 1;
        char *candidate = malloc(len);
        if (!candidate) {
            continue;
        }
        snprintf(candidate, len, "%s/%s", dir, word);
        if (access(candidate, X_OK) == 0) {
            found = 1;
            free(candidate);
            break;
        }
        free(candidate);
    }

    free(copy);
    return found;
}

static int clipboard_available(void) {
    return is_executable_word("wl-copy") ||
           is_executable_word("wl-paste") ||
           is_executable_word("xclip") ||
           is_executable_word("xsel");
}

static void shell_aliases_free(struct shell_aliases *aliases) {
    for (size_t i = 0; i < aliases->len; i++) {
        free(aliases->names[i]);
        free(aliases->values[i]);
    }
    free(aliases->names);
    free(aliases->values);
    aliases->names = NULL;
    aliases->values = NULL;
    aliases->len = 0;
    aliases->cap = 0;
}

static char *unquote_alias_value(const char *value) {
    value = skip_ws(value);
    size_t len = strlen(value);
    while (len > 0 && (value[len - 1] == '\n' || value[len - 1] == '\r')) {
        len--;
    }
    if (len >= 2 && (value[0] == '\'' || value[0] == '"') && value[len - 1] == value[0]) {
        char quote = value[0];
        char *out = malloc(len - 1);
        if (!out) {
            return NULL;
        }
        size_t n = 0;
        for (size_t i = 1; i + 1 < len; i++) {
            if (value[i] == '\\' && i + 2 < len && (value[i + 1] == quote || value[i + 1] == '\\')) {
                out[n++] = value[++i];
            } else {
                out[n++] = value[i];
            }
        }
        out[n] = '\0';
        return out;
    }
    char *out = malloc(len + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, value, len);
    out[len] = '\0';
    return out;
}

static int shell_aliases_add(struct shell_aliases *aliases, const char *name, size_t len, const char *value) {
    if (len == 0) {
        return 1;
    }
    for (size_t i = 0; i < aliases->len; i++) {
        if (strlen(aliases->names[i]) == len && strncmp(aliases->names[i], name, len) == 0) {
            return 1;
        }
    }
    if (aliases->len == aliases->cap) {
        size_t next_cap = aliases->cap ? aliases->cap * 2 : 32;
        char **next_names = malloc(next_cap * sizeof(*next_names));
        char **next_values = malloc(next_cap * sizeof(*next_values));
        if (!next_names || !next_values) {
            free(next_names);
            free(next_values);
            return 0;
        }
        for (size_t i = 0; i < aliases->len; i++) {
            next_names[i] = aliases->names[i];
            next_values[i] = aliases->values[i];
        }
        free(aliases->names);
        free(aliases->values);
        aliases->names = next_names;
        aliases->values = next_values;
        aliases->cap = next_cap;
    }
    aliases->names[aliases->len] = malloc(len + 1);
    aliases->values[aliases->len] = unquote_alias_value(value);
    if (!aliases->names[aliases->len] || !aliases->values[aliases->len]) {
        free(aliases->names[aliases->len]);
        free(aliases->values[aliases->len]);
        return 0;
    }
    memcpy(aliases->names[aliases->len], name, len);
    aliases->names[aliases->len][len] = '\0';
    aliases->len++;
    return 1;
}

static int shell_aliases_contains(const struct shell_aliases *aliases, const char *word) {
    if (!aliases || !word) {
        return 0;
    }
    for (size_t i = 0; i < aliases->len; i++) {
        if (strcmp(aliases->names[i], word) == 0) {
            return 1;
        }
    }
    return 0;
}

static const char *shell_aliases_value(const struct shell_aliases *aliases, const char *word) {
    if (!aliases || !word) {
        return NULL;
    }
    for (size_t i = 0; i < aliases->len; i++) {
        if (strcmp(aliases->names[i], word) == 0) {
            return aliases->values[i];
        }
    }
    return NULL;
}

static char *clean_alias_output_line(const char *line) {
    size_t len = strlen(line);
    char *out = malloc(len + 1);
    if (!out) {
        return NULL;
    }
    size_t n = 0;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)line[i];
        if (c == 27 && line[i + 1] == '[') {
            i += 2;
            while (line[i] && !isalpha((unsigned char)line[i])) {
                i++;
            }
            continue;
        }
        if ((c < 32 && c != '\n' && c != '\t') || c == 127) {
            continue;
        }
        out[n++] = (char)c;
    }
    out[n] = '\0';
    return out;
}

static void load_shell_aliases(struct shell_aliases *aliases) {
    const char *shell = getenv("SHELL");
    if (!shell || !*shell) {
        return;
    }
    char *quoted = shell_quote_word(shell);
    if (!quoted) {
        return;
    }
    size_t cmd_len = strlen(quoted) + 32;
    char *cmd = malloc(cmd_len);
    if (!cmd) {
        free(quoted);
        return;
    }
    snprintf(cmd, cmd_len, "%s -ic alias 2>/dev/null", quoted);
    free(quoted);

    FILE *fp = popen(cmd, "r");
    free(cmd);
    if (!fp) {
        return;
    }
    char *line = NULL;
    size_t cap = 0;
    while (getline(&line, &cap, fp) >= 0) {
        char *clean = clean_alias_output_line(line);
        if (!clean) {
            break;
        }
        char *p = clean;
        if (strncmp(p, "alias ", 6) == 0) {
            p += 6;
        }
        while (*p && isspace((unsigned char)*p)) {
            p++;
        }
        char *eq = strchr(p, '=');
        if (!eq || eq == p) {
            free(clean);
            continue;
        }
        size_t name_len = (size_t)(eq - p);
        if (!shell_aliases_add(aliases, p, name_len, eq + 1)) {
            free(clean);
            break;
        }
        free(clean);
    }
    free(line);
    pclose(fp);
}

static char *shell_word_after_first(const char *line) {
    line = skip_ws(line);
    if (!*line) {
        return NULL;
    }

    char quote = 0;
    const char *p = line;
    for (; *p; p++) {
        if (quote) {
            if (*p == quote) {
                quote = 0;
            } else if (*p == '\\' && p[1]) {
                p++;
            }
            continue;
        }
        if (*p == '\'' || *p == '"') {
            quote = *p;
        } else if (*p == '\\' && p[1]) {
            p++;
        } else if (isspace((unsigned char)*p)) {
            return (char *)skip_ws(p);
        }
    }
    return (char *)p;
}

static char *second_shell_word(const char *line) {
    char *tail = shell_word_after_first(line);
    if (!tail || !*tail) {
        return NULL;
    }
    return first_shell_word(tail);
}

static char *expand_alias_line(const char *line, const char *alias_value) {
    if (!alias_value) {
        return strdup(line);
    }
    const char *tail = shell_word_after_first(line);
    size_t len = strlen(alias_value) + (tail && *tail ? 1 + strlen(tail) : 0) + 1;
    char *out = malloc(len);
    if (!out) {
        return NULL;
    }
    if (tail && *tail) {
        snprintf(out, len, "%s %s", alias_value, tail);
    } else {
        snprintf(out, len, "%s", alias_value);
    }
    return out;
}

static int alias_name_seen(char **seen, size_t seen_len, const char *word) {
    for (size_t i = 0; i < seen_len; i++) {
        if (strcmp(seen[i], word) == 0) {
            return 1;
        }
    }
    return 0;
}

static char *expand_alias_line_recursive(const char *line, const struct shell_aliases *aliases) {
    char *current = strdup(line ? line : "");
    char *seen[64];
    size_t seen_len = 0;
    if (!current) {
        return NULL;
    }

    for (size_t depth = 0; depth < sizeof(seen) / sizeof(seen[0]); depth++) {
        char *word = first_shell_word(current);
        if (!word) {
            break;
        }
        const char *alias_value = shell_aliases_value(aliases, word);
        if (!alias_value) {
            free(word);
            break;
        }
        if (alias_name_seen(seen, seen_len, word)) {
            fprintf(stderr, "alias expansion stopped: recursive alias loop at %s\n", word);
            free(word);
            break;
        }
        seen[seen_len++] = word;

        char *next = expand_alias_line(current, alias_value);
        if (!next) {
            break;
        }
        free(current);
        current = next;
    }

    for (size_t i = 0; i < seen_len; i++) {
        free(seen[i]);
    }
    return current;
}

static int handle_blend_builtin(const char *line, const char *word, int *done) {
    if (strcmp(word, "exit") == 0 || strcmp(word, "logout") == 0 || strcmp(word, "quit") == 0) {
        *done = 1;
        return 1;
    }

    if (strcmp(word, "cd") == 0) {
        char *parsed_target = second_shell_word(line);
        const char *target = parsed_target;
        if (!target || !*target) {
            target = getenv("HOME");
        }
        if (!target || !*target) {
            fprintf(stderr, "cd: HOME is not set\n");
        } else if (chdir(target) != 0) {
            perror("cd");
        }
        free(parsed_target);
        return 1;
    }

    return 0;
}

static int should_execute_as_shell(const char *word, const struct shell_aliases *aliases) {
    return is_shell_reserved_or_builtin(word) || shell_aliases_contains(aliases, word) || is_executable_word(word);
}

static void command_result_clear(struct command_result *result) {
    free(result->command);
    free(result->direct_tty_name);
    free(result->output);
    result->command = NULL;
    result->direct_tty_name = NULL;
    result->output = NULL;
    result->exit_code = 0;
    result->valid = 0;
    result->direct_tty = 0;
}

static int command_result_set(struct command_result *result, const char *command, struct buffer *output, int exit_code) {
    command_result_clear(result);
    result->command = strdup(command);
    result->output = output->data ? output->data : strdup("");
    result->exit_code = exit_code;
    result->valid = result->command && result->output;
    output->data = NULL;
    output->len = 0;
    if (!result->valid) {
        command_result_clear(result);
        return 0;
    }
    return 1;
}

static int is_interactive_tty_command_word(const char *word) {
    static const char *commands[] = {
        "man", "less", "more", "most",
        "vi", "vim", "nvim", "nano", "micro", "emacs",
        "top", "htop", "btop", "atop", "glances",
        "watch", "tmux", "screen",
        "ssh", "sftp", "ftp",
        "codex", "hf"
    };

    if (!word || !*word) {
        return 0;
    }
    for (size_t i = 0; i < sizeof(commands) / sizeof(commands[0]); i++) {
        if (strcmp(word, commands[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

static int shell_line_needs_tty(const char *line) {
    char *word = first_shell_word(line);
    int needs_tty = is_interactive_tty_command_word(word);
    free(word);
    return needs_tty;
}

static int run_shell_line_direct(const char *line, struct command_result *last) {
    char *word = first_shell_word(line);
    command_result_clear(last);
    restore_terminal_if_needed();

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        free(word);
        return 1;
    }

    if (pid == 0) {
        execl("/bin/sh", "sh", "-c", line, (char *)NULL);
        perror("execl");
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        restore_terminal_if_needed();
        free(word);
        return 1;
    }
    restore_terminal_if_needed();

    if (WIFEXITED(status)) {
        last->command = strdup(line);
        last->direct_tty_name = word ? strdup(word) : strdup("");
        last->output = strdup("");
        last->exit_code = WEXITSTATUS(status);
        last->direct_tty = 1;
        last->valid = last->command && last->direct_tty_name && last->output;
        free(word);
        if (!last->valid) {
            command_result_clear(last);
            fprintf(stderr, "command result update failed: memory allocation failed\n");
        }
        return last->exit_code;
    }
    if (WIFSIGNALED(status)) {
        last->command = strdup(line);
        last->direct_tty_name = word ? strdup(word) : strdup("");
        last->output = strdup("");
        last->exit_code = 128 + WTERMSIG(status);
        last->direct_tty = 1;
        last->valid = last->command && last->direct_tty_name && last->output;
        free(word);
        if (!last->valid) {
            command_result_clear(last);
            fprintf(stderr, "command result update failed: memory allocation failed\n");
        }
        return last->exit_code;
    }
    free(word);
    return 1;
}

static int run_shell_line_capture(const char *line, struct command_result *last) {
    int fds[2];
    struct buffer output = {0};

    if (pipe(fds) != 0) {
        perror("pipe");
        return 1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        close(fds[0]);
        close(fds[1]);
        return 1;
    }

    if (pid == 0) {
        close(fds[0]);
        dup2(fds[1], STDOUT_FILENO);
        dup2(fds[1], STDERR_FILENO);
        close(fds[1]);
        execl("/bin/sh", "sh", "-c", line, (char *)NULL);
        perror("execl");
        _exit(127);
    }

    close(fds[1]);
    char chunk[4096];
    ssize_t nread;
    while ((nread = read(fds[0], chunk, sizeof(chunk))) > 0) {
        fwrite(chunk, 1, (size_t)nread, stdout);
        fflush(stdout);
        if (!append_bytes(&output, chunk, (size_t)nread)) {
            close(fds[0]);
            free(output.data);
            waitpid(pid, NULL, 0);
            fprintf(stderr, "command output capture failed: memory allocation failed\n");
            return 1;
        }
    }
    close(fds[0]);

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        perror("waitpid");
        free(output.data);
        return 1;
    }

    int exit_code = 1;
    if (WIFEXITED(status)) {
        exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        exit_code = 128 + WTERMSIG(status);
    }

    if (!command_result_set(last, line, &output, exit_code)) {
        free(output.data);
        fprintf(stderr, "command result capture failed: memory allocation failed\n");
    }
    return exit_code;
}

static int run_shell_line(const char *line, struct command_result *last) {
    if (isatty(STDIN_FILENO) && isatty(STDOUT_FILENO) && shell_line_needs_tty(line)) {
        return run_shell_line_direct(line, last);
    }
    return run_shell_line_capture(line, last);
}

static void print_direct_tty_notice(const struct command_result *last) {
    if (!last || !last->direct_tty) {
        return;
    }
    fprintf(stderr, "%s is a direct TTY command and thus its output was not captured\n",
            last->direct_tty_name && *last->direct_tty_name ? last->direct_tty_name : last->command);
}

static char *build_failure_question(const struct command_result *last) {
    const char *fmt =
        "the following command failed: '%s', with output of '%s', and exit code '%d', "
        "what is the problem, or solution?";
    size_t len = snprintf(NULL, 0, fmt, last->command, last->output, last->exit_code);
    char *question = malloc(len + 1);
    if (!question) {
        return NULL;
    }
    snprintf(question, len + 1, fmt, last->command, last->output, last->exit_code);
    return question;
}

static void code_blocks_clear(struct code_blocks *blocks) {
    for (size_t i = 0; i < blocks->len; i++) {
        free(blocks->items[i]);
    }
    free(blocks->items);
    blocks->items = NULL;
    blocks->len = 0;
    blocks->cap = 0;
}

static int code_blocks_add(struct code_blocks *blocks, const char *start, size_t len) {
    if (blocks->len == blocks->cap) {
        size_t next_cap = blocks->cap ? blocks->cap * 2 : 8;
        char **next = realloc(blocks->items, next_cap * sizeof(*next));
        if (!next) {
            return 0;
        }
        blocks->items = next;
        blocks->cap = next_cap;
    }

    char *copy = malloc(len + 1);
    if (!copy) {
        return 0;
    }
    memcpy(copy, start, len);
    copy[len] = '\0';
    blocks->items[blocks->len++] = copy;
    return 1;
}

static int is_shell_fence_lang(const char *start, size_t len) {
    while (len > 0 && isspace((unsigned char)*start)) {
        start++;
        len--;
    }

    char lang[16];
    size_t n = 0;
    while (n + 1 < sizeof(lang) && n < len &&
           !isspace((unsigned char)start[n]) && start[n] != '\r' && start[n] != '\n') {
        lang[n] = (char)tolower((unsigned char)start[n]);
        n++;
    }
    lang[n] = '\0';

    return strcmp(lang, "bash") == 0 ||
           strcmp(lang, "sh") == 0 ||
           strcmp(lang, "shell") == 0 ||
           strcmp(lang, "zsh") == 0;
}

static void extract_shell_code_blocks(const char *text, struct code_blocks *blocks) {
    const char *p = text;
    while ((p = strstr(p, "```")) != NULL) {
        const char *info_start = p + 3;
        const char *line_end = strchr(info_start, '\n');
        if (!line_end) {
            break;
        }

        int shell_lang = is_shell_fence_lang(info_start, (size_t)(line_end - info_start));
        const char *body_start = line_end + 1;
        const char *body_end = strstr(body_start, "```");
        if (!body_end) {
            break;
        }

        if (shell_lang) {
            size_t len = (size_t)(body_end - body_start);
            while (len > 0 && (body_start[len - 1] == '\n' || body_start[len - 1] == '\r')) {
                len--;
            }
            if (len > 0 && !code_blocks_add(blocks, body_start, len)) {
                fprintf(stderr, "code block capture failed: memory allocation failed\n");
                return;
            }
        }
        p = body_end + 3;
    }
}

static void print_numbered_code_blocks(const struct code_blocks *blocks, int color) {
    if (blocks->len == 0) {
        return;
    }

    if (color && isatty(STDOUT_FILENO)) {
        printf("\n%sExecutable shell blocks:%s\n", Q_COLOR_EXEC_BLOCK_LABEL, Q_COLOR_RESET);
    } else {
        printf("\nExecutable shell blocks:\n");
    }
    for (size_t i = 0; i < blocks->len; i++) {
        if (color && isatty(STDOUT_FILENO)) {
            printf("%s%zu.%s\n", Q_COLOR_EXEC_BLOCK_LABEL, i + 1, Q_COLOR_RESET);
        } else {
            printf("%zu.\n", i + 1);
        }
        printf("```bash\n%s\n```\n", blocks->items[i]);
    }
}

static int parse_block_reference(const char *line, size_t *index) {
    line = skip_ws(line);
    if (!isdigit((unsigned char)*line)) {
        return 0;
    }

    char *end = NULL;
    unsigned long value = strtoul(line, &end, 10);
    if (value == 0 || !end || *end != '.') {
        return 0;
    }
    end++;
    if (*skip_ws(end) != '\0') {
        return 0;
    }

    *index = (size_t)value - 1;
    return 1;
}

static void print_server_config_hint(const char *server, const char *port, const char *model, const char *api_key) {
    fprintf(stderr, "Check local server status and env var configuration.\n");
    fprintf(stderr, "Current env-derived values:\n");
    fprintf(stderr, "  LLM_SERVER=%s\n", server);
    fprintf(stderr, "  LLM_PORT=%s\n", port);
    fprintf(stderr, "  OPENAI_MODEL=%s\n", model);
    fprintf(stderr, "  OPENAI_API_KEY=%s\n", *api_key ? "<set>" : "");
}

static void print_llm_request_transport_error(CURLcode rc, const char *url,
                                              const char *server, const char *port,
                                              const char *model, const char *api_key,
                                              long http_status) {
    fprintf(stderr, "LLM request failed: %s\n", curl_easy_strerror(rc));
    fprintf(stderr, "URL: %s\n", url ? url : "");
    if (http_status > 0) {
        fprintf(stderr, "HTTP status before failure: %ld\n", http_status);
    }

    switch (rc) {
        case CURLE_GOT_NOTHING:
            fprintf(stderr,
                    "Likely cause: the server accepted the connection but closed it without sending an HTTP response.\n"
                    "Check whether the LLM process crashed, restarted, ran out of memory, rejected the route, or is not actually serving OpenAI-compatible /v1/responses on this port.\n");
            break;
        case CURLE_OPERATION_TIMEDOUT:
            fprintf(stderr,
                    "Likely cause: the server did not respond before the timeout.\n"
                    "Check --llm-timeout, /llm-timeout, server load, model load time, and whether streaming responses are blocked.\n");
            break;
        case CURLE_RECV_ERROR:
            fprintf(stderr,
                    "Likely cause: the connection was interrupted while q was receiving the response.\n"
                    "Check server logs for a crash, context overflow, proxy reset, or model backend failure.\n");
            break;
        case CURLE_SEND_ERROR:
            fprintf(stderr,
                    "Likely cause: the connection was interrupted while q was sending the request.\n"
                    "Check server logs, request size, and local network/proxy stability.\n");
            break;
        case CURLE_HTTP2:
        case CURLE_HTTP2_STREAM:
            fprintf(stderr,
                    "Likely cause: the HTTP/2 stream was reset or the server/proxy has an HTTP/2 compatibility problem.\n"
                    "Try the server directly without a proxy, or configure the server/proxy for HTTP/1.1 compatibility.\n");
            break;
        case CURLE_SSL_CONNECT_ERROR:
        case CURLE_PEER_FAILED_VERIFICATION:
            fprintf(stderr,
                    "Likely cause: TLS setup or certificate verification failed.\n"
                    "Check whether LLM_SERVER uses the correct http:// or https:// scheme and certificate configuration.\n");
            break;
        default:
            fprintf(stderr,
                    "Check the LLM server logs, route compatibility, request size, model availability, and API logging output.\n");
            break;
    }
    print_server_config_hint(server, port, model, api_key);
}

static const char *env_or_default(const char *name, const char *default_value) {
    const char *value = getenv(name);
    return value && *value ? value : default_value;
}

static char *path_join2(const char *a, const char *b) {
    size_t a_len = strlen(a);
    size_t b_len = strlen(b);
    int slash = a_len > 0 && a[a_len - 1] != '/';
    char *out = malloc(a_len + (slash ? 1 : 0) + b_len + 1);
    if (!out) {
        return NULL;
    }
    snprintf(out, a_len + (slash ? 1 : 0) + b_len + 1, "%s%s%s", a, slash ? "/" : "", b);
    return out;
}

static int ensure_dir(const char *path);

static char *config_home_dir_path(void) {
    const char *home = getenv("HOME");
    if (!home || !*home) {
        home = ".";
    }
    return path_join2(home, ".config");
}

static int ensure_config_home_dir(char **out_dir) {
    char *config = config_home_dir_path();
    if (!config) {
        return 0;
    }
    if (!ensure_dir(config)) {
        free(config);
        return 0;
    }
    *out_dir = config;
    return 1;
}

static char *config_child_path(const char *name) {
    char *config = config_home_dir_path();
    if (!config) {
        return NULL;
    }
    char *path = path_join2(config, name);
    free(config);
    return path;
}

static char *q_config_dir_path(void) {
    return config_child_path("q");
}

static int ensure_q_config_dir(char **out_dir) {
    char *config = NULL;
    if (!ensure_config_home_dir(&config)) {
        return 0;
    }
    char *qdir = path_join2(config, "q");
    free(config);
    if (!qdir) {
        return 0;
    }
    if (!ensure_dir(qdir)) {
        free(qdir);
        return 0;
    }
    *out_dir = qdir;
    return 1;
}

static char *q_config_child_path(const char *name, int create_dir) {
    char *dir = NULL;
    if (create_dir) {
        if (!ensure_q_config_dir(&dir)) {
            return NULL;
        }
    } else {
        dir = q_config_dir_path();
        if (!dir) {
            return NULL;
        }
    }
    char *path = path_join2(dir, name);
    free(dir);
    return path;
}

static char *sessions_dir_path(void) {
    return q_config_child_path("sessions", 0);
}

static char *mcp_config_path(int create_dir) {
    return q_config_child_path("mcp-servers.tsv", create_dir);
}

static char *notes_file_path(int create_dir) {
    return q_config_child_path("NOTES", create_dir);
}

static int append_note_line(const char *text) {
    char *path = notes_file_path(1);
    if (!path) {
        fprintf(stderr, "/note: failed to resolve notes path\n");
        return 0;
    }

    FILE *fp = fopen(path, "a");
    if (!fp) {
        fprintf(stderr, "/note: cannot append to %s: %s\n", path, strerror(errno));
        free(path);
        return 0;
    }
    int write_failed = fprintf(fp, "%s\n", text ? text : "") < 0;
    int close_failed = fclose(fp) != 0;
    if (write_failed || close_failed) {
        fprintf(stderr, "/note: failed to write %s: %s\n", path, strerror(errno));
        free(path);
        return 0;
    }
    fprintf(stderr, "/note: appended to %s\n", path);
    free(path);
    return 1;
}

static int ensure_dir(const char *path) {
    if (mkdir(path, 0700) == 0 || errno == EEXIST) {
        return 1;
    }
    perror(path);
    return 0;
}

static int ensure_sessions_dir(char **out_dir) {
    char *qdir = NULL;
    char *sessions = NULL;
    if (!ensure_q_config_dir(&qdir)) {
        return 0;
    }
    sessions = path_join2(qdir, "sessions");
    free(qdir);
    if (!sessions) {
        return 0;
    }
    if (!ensure_dir(sessions)) {
        free(sessions);
        return 0;
    }
    *out_dir = sessions;
    return 1;
}

static int ensure_named_config_dir(const char *name, char **out_dir) {
    char *config = NULL;
    char *dir = NULL;
    if (!ensure_config_home_dir(&config)) {
        return 0;
    }
    dir = path_join2(config, name);
    free(config);
    if (!dir) {
        return 0;
    }
    if (!ensure_dir(dir)) {
        free(dir);
        return 0;
    }
    *out_dir = dir;
    return 1;
}

static void mcp_registry_free(struct mcp_registry *reg) {
    for (size_t i = 0; i < reg->server_len; i++) {
        free(reg->servers[i].name);
        free(reg->servers[i].command);
    }
    for (size_t i = 0; i < reg->tool_len; i++) {
        free(reg->tools[i].server);
        free(reg->tools[i].tool);
        free(reg->tools[i].function);
        free(reg->tools[i].description);
        free(reg->tools[i].schema);
    }
    free(reg->servers);
    free(reg->tools);
    memset(reg, 0, sizeof(*reg));
}

static int mcp_add_server_mem(struct mcp_registry *reg, const char *name, const char *url) {
    if (!name || !*name || !url || !*url) {
        return 0;
    }
    for (size_t i = 0; i < reg->server_len; i++) {
        if (strcmp(reg->servers[i].name, name) == 0) {
            char *next = strdup(url);
            if (!next) {
                return 0;
            }
            free(reg->servers[i].command);
            reg->servers[i].command = next;
            reg->servers[i].alive = 0;
            reg->servers[i].tool_count = 0;
            return 1;
        }
    }
    if (reg->server_len == reg->server_cap) {
        size_t next_cap = reg->server_cap ? reg->server_cap * 2 : 8;
        struct mcp_server *next = realloc(reg->servers, next_cap * sizeof(*next));
        if (!next) {
            return 0;
        }
        reg->servers = next;
        reg->server_cap = next_cap;
    }
    reg->servers[reg->server_len].name = strdup(name);
    reg->servers[reg->server_len].command = strdup(url);
    reg->servers[reg->server_len].alive = 0;
    reg->servers[reg->server_len].tool_count = 0;
    if (!reg->servers[reg->server_len].name || !reg->servers[reg->server_len].command) {
        free(reg->servers[reg->server_len].name);
        free(reg->servers[reg->server_len].command);
        return 0;
    }
    reg->server_len++;
    return 1;
}

static int mcp_load_servers(struct mcp_registry *reg) {
    char *path = mcp_config_path(0);
    if (!path) {
        return 0;
    }
    FILE *fp = fopen(path, "r");
    free(path);
    if (!fp) {
        return 1;
    }
    char *line = NULL;
    size_t cap = 0;
    int ok = 1;
    while (getline(&line, &cap, fp) >= 0) {
        char *nl = strpbrk(line, "\r\n");
        if (nl) {
            *nl = '\0';
        }
        if (!*line || line[0] == '#') {
            continue;
        }
        char *tab = strchr(line, '\t');
        if (!tab) {
            continue;
        }
        *tab = '\0';
        if (!mcp_add_server_mem(reg, line, tab + 1)) {
            ok = 0;
            break;
        }
    }
    free(line);
    fclose(fp);
    return ok;
}

static int mcp_save_servers(const struct mcp_registry *reg) {
    char *path = mcp_config_path(1);
    if (!path) {
        return 0;
    }
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror(path);
        free(path);
        return 0;
    }
    int ok = 1;
    for (size_t i = 0; i < reg->server_len; i++) {
        if (fprintf(fp, "%s\t%s\n", reg->servers[i].name, reg->servers[i].command) < 0) {
            ok = 0;
            break;
        }
    }
    if (fclose(fp) != 0) {
        perror(path);
        ok = 0;
    }
    free(path);
    return ok;
}

static int mcp_remove_server_mem(struct mcp_registry *reg, const char *name) {
    for (size_t i = 0; i < reg->server_len; i++) {
        if (strcmp(reg->servers[i].name, name) == 0) {
            free(reg->servers[i].name);
            free(reg->servers[i].command);
            memmove(reg->servers + i, reg->servers + i + 1, (reg->server_len - i - 1) * sizeof(*reg->servers));
            reg->server_len--;
            return 1;
        }
    }
    return 0;
}

static size_t mcp_curl_write_cb(char *ptr, size_t size, size_t nmemb, void *userdata) {
    size_t n = size * nmemb;
    struct buffer *buf = userdata;
    return append_bytes(buf, ptr, n) ? n : 0;
}

static char *mcp_extract_json_response(const char *raw) {
    if (!raw) {
        return NULL;
    }
    const char *p = skip_ws(raw);
    if (*p == '{') {
        return strdup(p);
    }
    while ((p = strstr(p, "data:")) != NULL) {
        p = skip_ws(p + 5);
        if (strncmp(p, "[DONE]", 6) == 0) {
            p += 6;
            continue;
        }
        if (*p == '{') {
            const char *end = strchr(p, '\n');
            size_t len = end ? (size_t)(end - p) : strlen(p);
            while (len > 0 && (p[len - 1] == '\r' || p[len - 1] == '\n')) {
                len--;
            }
            char *out = malloc(len + 1);
            if (!out) {
                return NULL;
            }
            memcpy(out, p, len);
            out[len] = '\0';
            return out;
        }
    }
    return NULL;
}

static char *mcp_http_post_json(const char *url, const char *payload) {
    CURL *curl = curl_easy_init();
    if (!curl) {
        return NULL;
    }
    struct buffer raw = {0};
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: application/json, text/event-stream");

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, mcp_curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &raw);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "q/1.0");

    CURLcode rc = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    if (rc != CURLE_OK || status < 200 || status >= 300) {
        free(raw.data);
        return NULL;
    }
    char *json = mcp_extract_json_response(raw.data ? raw.data : "");
    free(raw.data);
    return json;
}

static char *mcp_rpc_request(const char *url, long id, const char *method, const char *params_json) {
    char *method_json = json_escape(method);
    if (!method_json) {
        return NULL;
    }
    const char *params = params_json ? params_json : "{}";
    size_t len = snprintf(NULL, 0,
                          "{\"jsonrpc\":\"2.0\",\"id\":%ld,\"method\":%s,\"params\":%s}",
                          id, method_json, params);
    char *payload = malloc(len + 1);
    if (!payload) {
        free(method_json);
        return NULL;
    }
    snprintf(payload, len + 1,
             "{\"jsonrpc\":\"2.0\",\"id\":%ld,\"method\":%s,\"params\":%s}",
             id, method_json, params);
    free(method_json);
    char *response = mcp_http_post_json(url, payload);
    free(payload);
    return response;
}

static void mcp_rpc_notify_initialized(const char *url) {
    (void)mcp_http_post_json(url, "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\",\"params\":{}}");
}

static char *mcp_server_name_from_url(const char *url) {
    const char *init_params =
        "{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},\"clientInfo\":{\"name\":\"q\",\"version\":\"1.0\"}}";
    char *init = mcp_rpc_request(url, 1, "initialize", init_params);
    if (!init) {
        return NULL;
    }
    char *server_info = get_json_value(init, "serverInfo");
    char *name = server_info ? get_json_string(server_info, "name") : NULL;
    if (!name) {
        name = get_json_string(init, "name");
    }
    free(server_info);
    free(init);
    if (!name || !*name) {
        free(name);
        return NULL;
    }
    return name;
}

static char *mcp_sanitize_name(const char *server, const char *tool) {
    size_t len = strlen("mcp__") + strlen(server) + 2 + strlen(tool) + 1;
    char *out = malloc(len);
    if (!out) {
        return NULL;
    }
    snprintf(out, len, "mcp__%s__%s", server, tool);
    for (char *p = out; *p; p++) {
        if (!isalnum((unsigned char)*p) && *p != '_') {
            *p = '_';
        }
    }
    return out;
}

static int mcp_add_tool(struct mcp_registry *reg, const char *server, const char *tool, const char *description, const char *schema) {
    if (reg->tool_len == reg->tool_cap) {
        size_t next_cap = reg->tool_cap ? reg->tool_cap * 2 : 16;
        struct mcp_tool *next = realloc(reg->tools, next_cap * sizeof(*next));
        if (!next) {
            return 0;
        }
        reg->tools = next;
        reg->tool_cap = next_cap;
    }
    struct mcp_tool *t = &reg->tools[reg->tool_len];
    t->server = strdup(server);
    t->tool = strdup(tool);
    t->function = mcp_sanitize_name(server, tool);
    t->description = strdup(description && *description ? description : "MCP tool");
    t->schema = strdup(schema && *skip_ws(schema) == '{' ? schema : "{\"type\":\"object\",\"properties\":{}}");
    if (!t->server || !t->tool || !t->function || !t->description || !t->schema) {
        free(t->server);
        free(t->tool);
        free(t->function);
        free(t->description);
        free(t->schema);
        return 0;
    }
    reg->tool_len++;
    return 1;
}

static const char *json_next_object(const char *p, char **out) {
    while (p && *p && *p != '{') {
        p++;
    }
    if (!p || !*p) {
        return NULL;
    }
    *out = parse_json_value_at(p);
    if (!*out) {
        return NULL;
    }
    return p + strlen(*out);
}

static int mcp_parse_tools_response(struct mcp_registry *reg, struct mcp_server *server, const char *response) {
    char *tools = get_json_value(response, "tools");
    if (!tools) {
        return 0;
    }
    const char *p = tools;
    int added = 0;
    char *obj = NULL;
    while ((p = json_next_object(p, &obj)) != NULL) {
        char *name = get_json_string(obj, "name");
        char *desc = get_json_string(obj, "description");
        char *schema = get_json_value(obj, "inputSchema");
        if (name && *name && mcp_add_tool(reg, server->name, name, desc, schema)) {
            added++;
        }
        free(name);
        free(desc);
        free(schema);
        free(obj);
        obj = NULL;
    }
    free(tools);
    server->tool_count = (size_t)added;
    return added >= 0;
}

static int mcp_probe_server(struct mcp_registry *reg, struct mcp_server *server) {
    const char *init_params =
        "{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},\"clientInfo\":{\"name\":\"q\",\"version\":\"1.0\"}}";
    char *init = mcp_rpc_request(server->command, 1, "initialize", init_params);
    if (!init) {
        server->alive = 0;
        return 0;
    }
    free(init);
    mcp_rpc_notify_initialized(server->command);
    char *list = mcp_rpc_request(server->command, 2, "tools/list", "{}");
    if (!list) {
        server->alive = 0;
        return 0;
    }
    server->alive = 1;
    int ok = mcp_parse_tools_response(reg, server, list);
    free(list);
    return ok;
}

static void mcp_probe_all(struct mcp_registry *reg) {
    for (size_t i = 0; i < reg->server_len; i++) {
        mcp_probe_server(reg, &reg->servers[i]);
    }
}

static int mcp_list_servers_command(void) {
    mcp_registry_free(&mcp_registry);
    if (!mcp_load_servers(&mcp_registry)) {
        fprintf(stderr, "failed to load MCP server config\n");
        return 1;
    }
    mcp_probe_all(&mcp_registry);
    printf("%-20s %-8s %5s %s\n", "Name", "Alive", "Tools", "URL");
    for (size_t i = 0; i < mcp_registry.server_len; i++) {
        printf("%-20s %-8s %5zu %s\n",
               mcp_registry.servers[i].name,
               mcp_registry.servers[i].alive ? "yes" : "no",
               mcp_registry.servers[i].tool_count,
               mcp_registry.servers[i].command);
    }
    return 0;
}

static int mcp_add_server_command(const char *url) {
    if (!url || !*url) {
        fprintf(stderr, "--add-mcp-server: missing url\n");
        return 1;
    }
    char *derived_name = mcp_server_name_from_url(url);
    if (!derived_name) {
        fprintf(stderr, "--add-mcp-server: failed to read server name from initialize response\n");
        return 1;
    }
    if (!mcp_add_server_mem(&mcp_registry, derived_name, url) || !mcp_save_servers(&mcp_registry)) {
        fprintf(stderr, "--add-mcp-server failed\n");
        free(derived_name);
        return 1;
    }
    printf("MCP server added: %s %s\n", derived_name, url);
    free(derived_name);
    return 0;
}

static int mcp_remove_server_command(const char *name) {
    if (!name || !*name) {
        fprintf(stderr, "--remove-mcp-server: missing name\n");
        return 1;
    }
    if (!mcp_remove_server_mem(&mcp_registry, name)) {
        fprintf(stderr, "--remove-mcp-server: no such server: %s\n", name);
        return 1;
    }
    if (!mcp_save_servers(&mcp_registry)) {
        fprintf(stderr, "--remove-mcp-server failed\n");
        return 1;
    }
    printf("MCP server removed: %s\n", name);
    return 0;
}

static struct mcp_tool *mcp_find_tool_by_function(const char *function) {
    for (size_t i = 0; i < mcp_registry.tool_len; i++) {
        if (strcmp(mcp_registry.tools[i].function, function) == 0) {
            return &mcp_registry.tools[i];
        }
    }
    return NULL;
}

static const char *mcp_server_url(const char *server_name) {
    for (size_t i = 0; i < mcp_registry.server_len; i++) {
        if (strcmp(mcp_registry.servers[i].name, server_name) == 0) {
            return mcp_registry.servers[i].command;
        }
    }
    return NULL;
}

static const char *skip_json_string_raw(const char *p) {
    if (!p || *p != '"') {
        return p;
    }
    p++;
    int escaped = 0;
    while (*p) {
        if (escaped) {
            escaped = 0;
        } else if (*p == '\\') {
            escaped = 1;
        } else if (*p == '"') {
            return p + 1;
        }
        p++;
    }
    return p;
}

static void append_tool_display_value(struct buffer *buf, const char *value, int is_string) {
    size_t shown = 0;
    int last_space = 0;
    const size_t max = is_string ? 100 : 0;
    const char *text = value ? value : "";
    size_t value_len = strlen(text);
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        char c = (char)*p;
        if (isspace(*p)) {
            if (last_space) {
                continue;
            }
            c = ' ';
            last_space = 1;
        } else {
            last_space = 0;
        }
        if (max > 0 && shown >= max) {
            char suffix[64];
            snprintf(suffix, sizeof(suffix), "... (%zu chars remaining)", value_len - max);
            append_bytes(buf, suffix, strlen(suffix));
            return;
        }
        append_bytes(buf, &c, 1);
        shown++;
    }
}

static char *tool_arguments_summary(const char *arguments) {
    struct buffer out = {0};
    const char *p = skip_ws(arguments ? arguments : "");
    if (*p != '{') {
        return strdup("{}");
    }
    p++;
    append_bytes(&out, "{ ", 2);
    int count = 0;
    while (*p) {
        p = skip_ws(p);
        if (*p == '}') {
            break;
        }
        if (*p != '"') {
            break;
        }
        char *key = parse_json_string_at(p);
        p = skip_json_string_raw(p);
        p = skip_ws(p);
        if (!key || *p != ':') {
            free(key);
            break;
        }
        p = skip_ws(p + 1);

        char *value = NULL;
        int value_is_string = 0;
        if (*p == '"') {
            value_is_string = 1;
            value = parse_json_string_at(p);
            p = skip_json_string_raw(p);
        } else if (*p == '{' || *p == '[') {
            value = parse_json_value_at(p);
            if (value) {
                p += strlen(value);
            }
        } else {
            const char *start = p;
            while (*p && *p != ',' && *p != '}') {
                p++;
            }
            const char *end = p;
            while (end > start && isspace((unsigned char)end[-1])) {
                end--;
            }
            size_t len = (size_t)(end - start);
            value = malloc(len + 1);
            if (value) {
                memcpy(value, start, len);
                value[len] = '\0';
            }
        }

        if (count > 0) {
            append_bytes(&out, ", ", 2);
        }
        append_bytes(&out, key, strlen(key));
        append_bytes(&out, " : ", 3);
        append_tool_display_value(&out, value ? value : "", value_is_string);
        count++;

        free(key);
        free(value);
        p = skip_ws(p);
        if (*p == ',') {
            p++;
            continue;
        }
        break;
    }
    if (count == 0) {
        append_bytes(&out, "}", 1);
    } else {
        append_bytes(&out, " }", 2);
    }
    if (!out.data) {
        return strdup("{}");
    }
    return out.data;
}

static void print_tool_call_line(const char *function, const char *arguments, int loud) {
    struct mcp_tool *tool = mcp_find_tool_by_function(function);
    const char *display_name = tool ? tool->tool : (function ? function : "");
    if (!loud) {
        fprintf(stderr, "[Tool-Call: %s]\n", display_name);
        fflush(stderr);
        return;
    }

    char *summary = tool_arguments_summary(arguments);
    if (tool) {
        const char *url = mcp_server_url(tool->server);
        fprintf(stderr, "[Tool-Call: %s (%s@%s)] %s\n",
                tool->tool, tool->server, url ? url : "", summary ? summary : "{}");
    } else {
        fprintf(stderr, "[Tool-Call: %s] %s\n", display_name, summary ? summary : "{}");
    }
    fflush(stderr);
    free(summary);
}

static char *execute_mcp_tool_call(const char *function, const char *arguments) {
    struct mcp_tool *tool = mcp_find_tool_by_function(function);
    if (!tool) {
        return NULL;
    }
    const char *url = mcp_server_url(tool->server);
    if (!url) {
        return strdup("{\"ok\":false,\"error\":\"MCP server not found\"}");
    }
    char *tool_name = json_escape(tool->tool);
    if (!tool_name) {
        return strdup("{\"ok\":false,\"error\":\"memory allocation failed\"}");
    }
    const char *args = arguments && *skip_ws(arguments) == '{' ? arguments : "{}";
    size_t params_len = snprintf(NULL, 0, "{\"name\":%s,\"arguments\":%s}", tool_name, args);
    char *params = malloc(params_len + 1);
    if (!params) {
        free(tool_name);
        return strdup("{\"ok\":false,\"error\":\"memory allocation failed\"}");
    }
    snprintf(params, params_len + 1, "{\"name\":%s,\"arguments\":%s}", tool_name, args);
    free(tool_name);

    const char *init_params =
        "{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},\"clientInfo\":{\"name\":\"q\",\"version\":\"1.0\"}}";
    char *init = mcp_rpc_request(url, 1, "initialize", init_params);
    if (!init) {
        free(params);
        return strdup("{\"ok\":false,\"error\":\"MCP initialize failed\"}");
    }
    free(init);
    mcp_rpc_notify_initialized(url);

    char *response = mcp_rpc_request(url, 2, "tools/call", params);
    free(params);
    if (!response) {
        return strdup("{\"ok\":false,\"error\":\"MCP tools/call failed\"}");
    }
    return response;
}

static int append_tool_definitions(struct buffer *buf) {
    const char *builtin =
        ",\"tools\":["
        "{\"type\":\"function\",\"name\":\"read_file\",\"description\":\"Read a local text file from the user's machine.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"Path of the file to read.\"}},\"required\":[\"path\"],\"additionalProperties\":false}},"
        "{\"type\":\"function\",\"name\":\"write_file\",\"description\":\"Write text content to a local file on the user's machine.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"Path of the file to write.\"},\"content\":{\"type\":\"string\",\"description\":\"Complete file content to write.\"}},\"required\":[\"path\",\"content\"],\"additionalProperties\":false}},"
        "{\"type\":\"function\",\"name\":\"get_time\",\"description\":\"Return the current time from the user's machine, including local time, UTC time, timezone, UTC offset, and Unix timestamp.\",\"parameters\":{\"type\":\"object\",\"properties\":{},\"additionalProperties\":false}}";
    if (!append_bytes(buf, builtin, strlen(builtin))) {
        return 0;
    }

    for (size_t i = 0; i < mcp_registry.tool_len; i++) {
        struct mcp_tool *tool = &mcp_registry.tools[i];
        char *name = json_escape(tool->function);
        char *desc_raw = NULL;
        size_t desc_raw_len = strlen("[MCP ] ") + strlen(tool->server) + strlen(tool->description) + 1;
        desc_raw = malloc(desc_raw_len);
        if (desc_raw) {
            snprintf(desc_raw, desc_raw_len, "[MCP %s] %s", tool->server, tool->description);
        }
        char *desc = json_escape(desc_raw ? desc_raw : "MCP tool");
        free(desc_raw);
        if (!name || !desc) {
            free(name);
            free(desc);
            return 0;
        }
        if (!append_bytes(buf, ",{\"type\":\"function\",\"name\":", strlen(",{\"type\":\"function\",\"name\":")) ||
            !append_bytes(buf, name, strlen(name)) ||
            !append_bytes(buf, ",\"description\":", strlen(",\"description\":")) ||
            !append_bytes(buf, desc, strlen(desc)) ||
            !append_bytes(buf, ",\"parameters\":", strlen(",\"parameters\":")) ||
            !append_bytes(buf, tool->schema, strlen(tool->schema)) ||
            !append_bytes(buf, "}", 1)) {
            free(name);
            free(desc);
            return 0;
        }
        free(name);
        free(desc);
    }
    return append_bytes(buf, "]", 1);
}

static char *read_text_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        return NULL;
    }

    struct buffer out = {0};
    char chunk[4096];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), fp)) > 0) {
        if (!append_bytes(&out, chunk, n)) {
            free(out.data);
            fclose(fp);
            return NULL;
        }
    }
    fclose(fp);
    if (!out.data) {
        return strdup("");
    }
    return out.data;
}

static int write_text_file(const char *path, const char *text) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        return 0;
    }
    if (fputs(text ? text : "", fp) == EOF) {
        fclose(fp);
        return 0;
    }
    return fclose(fp) == 0;
}

static char *default_system_prompt_path(void) {
    return q_config_child_path("SYSTEM.txt", 0);
}

static int set_system_prompt_file(const char *path, int quiet) {
    if (!path || !*path) {
        if (!quiet) {
            fprintf(stderr, "system prompt: missing file path\n");
        }
        return 0;
    }
    struct stat st;
    if (stat(path, &st) != 0) {
        if (!quiet) {
            fprintf(stderr, "system prompt: %s: %s\n", path, strerror(errno));
        }
        return 0;
    }
    if (!S_ISREG(st.st_mode)) {
        if (!quiet) {
            fprintf(stderr, "system prompt: %s is not a regular file\n", path);
        }
        return 0;
    }
    if (access(path, R_OK) != 0) {
        if (!quiet) {
            fprintf(stderr, "system prompt: %s is not readable: %s\n", path, strerror(errno));
        }
        return 0;
    }
    char *text = read_text_file(path);
    if (!text) {
        if (!quiet) {
            fprintf(stderr, "system prompt: failed to read %s\n", path);
        }
        return 0;
    }
    char *copy = strdup(path);
    if (!copy) {
        free(text);
        if (!quiet) {
            fprintf(stderr, "system prompt: memory allocation failed\n");
        }
        return 0;
    }
    free(system_prompt_text);
    free(system_prompt_path);
    system_prompt_text = text;
    system_prompt_path = copy;
    return 1;
}

static void load_default_system_prompt_if_available(void) {
    if (system_prompt_text) {
        return;
    }
    char *path = default_system_prompt_path();
    if (!path) {
        return;
    }
    if (access(path, F_OK) == 0) {
        set_system_prompt_file(path, 1);
    }
    free(path);
}

static char *safe_cache_name(const char *cmd) {
    const char *base = strrchr(cmd, '/');
    base = base ? base + 1 : cmd;
    if (!base || !*base) {
        base = "command";
    }

    size_t len = strlen(base);
    char *out = malloc(len + 5);
    if (!out) {
        return NULL;
    }
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)base[i];
        out[i] = (isalnum(c) || c == '_' || c == '-' || c == '.') ? (char)c : '_';
    }
    memcpy(out + len, ".txt", 5);
    return out;
}

static char *completion_cache_path(const char *cmd, int create_dir) {
    char *dir = NULL;
    char *name = NULL;
    char *path = NULL;

    if (create_dir) {
        if (!ensure_named_config_dir("q-completions", &dir)) {
            return NULL;
        }
    } else {
        dir = config_child_path("q-completions");
        if (!dir) {
            return NULL;
        }
    }

    name = safe_cache_name(cmd);
    if (name) {
        path = path_join2(dir, name);
    }
    free(dir);
    free(name);
    return path;
}

static char *completion_cache_dir_path(void) {
    return config_child_path("q-completions");
}

static int clear_completion_cache(void) {
    char *dir = completion_cache_dir_path();
    if (!dir) {
        fprintf(stderr, "completion cache: memory allocation failed\n");
        return 1;
    }

    DIR *dp = opendir(dir);
    if (!dp) {
        if (errno == ENOENT) {
            printf("completion cache: already empty\n");
            free(dir);
            return 0;
        }
        perror("completion cache opendir");
        free(dir);
        return 1;
    }

    int removed = 0;
    int failed = 0;
    struct dirent *ent;
    while ((ent = readdir(dp)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }
        size_t n = strlen(ent->d_name);
        if (n < 5 || strcmp(ent->d_name + n - 4, ".txt") != 0) {
            continue;
        }

        char *path = path_join2(dir, ent->d_name);
        if (!path) {
            failed = 1;
            continue;
        }
        if (unlink(path) == 0) {
            removed++;
        } else {
            perror(path);
            failed = 1;
        }
        free(path);
    }
    closedir(dp);
    free(dir);

    printf("completion cache: removed %d entr%s\n", removed, removed == 1 ? "y" : "ies");
    return failed ? 1 : 0;
}

static int is_sequence_resume_id(const char *id);
static char *resume_session_path_by_sequence(const char *id);
static char *resume_session_path_last(void);

static void free_string_array(char **items, size_t len) {
    for (size_t i = 0; i < len; i++) {
        free(items[i]);
    }
    free(items);
}

static char *new_session_path(void) {
    char *dir = NULL;
    if (!ensure_sessions_dir(&dir)) {
        return NULL;
    }

    time_t now = time(NULL);
    struct tm tm_now;
    localtime_r(&now, &tm_now);

    char name[64];
    strftime(name, sizeof(name), "session-%Y%m%d%H%M%S", &tm_now);
    char *path = path_join2(dir, name);
    free(dir);
    return path;
}

static char *resume_session_path(const char *id) {
    if (strcmp(id, "last") == 0) {
        return resume_session_path_last();
    }
    if (is_sequence_resume_id(id)) {
        return resume_session_path_by_sequence(id);
    }

    char *dir = NULL;
    if (!ensure_sessions_dir(&dir)) {
        return NULL;
    }

    const char *name = strncmp(id, "session-", 8) == 0 ? id : NULL;
    char prefixed[128];
    if (!name) {
        snprintf(prefixed, sizeof(prefixed), "session-%s", id);
        name = prefixed;
    }

    char *path = path_join2(dir, name);
    free(dir);
    return path;
}

static long count_file_lines(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        return -1;
    }
    long lines = 0;
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (c == '\n') {
            lines++;
        }
    }
    fclose(fp);
    return lines;
}

static void format_age(time_t then, char *buf, size_t len) {
    time_t now = time(NULL);
    long seconds = (long)difftime(now, then);
    if (seconds < 0) {
        seconds = 0;
    }
    long mins = seconds / 60;
    if (mins < 60) {
        snprintf(buf, len, "%5ld m", mins);
        return;
    }
    long hours = mins / 60;
    if (hours < 48) {
        snprintf(buf, len, "%5ld h", hours);
        return;
    }
    long days = hours / 24;
    snprintf(buf, len, "%5ld d", days);
}

static int session_name_cmp(const void *a, const void *b) {
    const char *const *sa = a;
    const char *const *sb = b;
    return strcmp(*sa, *sb);
}

static int is_sequence_resume_id(const char *id);
static char *resume_session_path_by_sequence(const char *id);
static char *resume_session_path_last(void);

static int is_sequence_resume_id(const char *id) {
    if (!id || !*id) {
        return 0;
    }
    size_t len = strlen(id);
    if (len >= 8) {
        return 0;
    }
    for (const unsigned char *p = (const unsigned char *)id; *p; p++) {
        if (!isdigit(*p)) {
            return 0;
        }
    }
    return strtol(id, NULL, 10) > 0;
}

static int load_session_names(int create_dir, int missing_ok, char ***out_names, size_t *out_len, char **out_dir) {
    char *dir = NULL;
    char **names = NULL;
    size_t len = 0;
    size_t cap = 0;
    *out_names = NULL;
    *out_len = 0;
    if (out_dir) {
        *out_dir = NULL;
    }
    if (create_dir) {
        if (!ensure_sessions_dir(&dir)) {
            return 0;
        }
    } else {
        dir = sessions_dir_path();
        if (!dir) {
            return 0;
        }
    }
    DIR *dp = opendir(dir);
    if (!dp) {
        int missing = errno == ENOENT;
        if (!missing || !missing_ok) {
            perror(dir);
        }
        free(dir);
        return missing && missing_ok;
    }

    struct dirent *ent;
    while ((ent = readdir(dp)) != NULL) {
        if (strncmp(ent->d_name, "session-", 8) != 0) {
            continue;
        }
        if (len == cap) {
            size_t next_cap = cap ? cap * 2 : 16;
            char **next = realloc(names, next_cap * sizeof(*next));
            if (!next) {
                closedir(dp);
                free_string_array(names, len);
                free(dir);
                return 0;
            }
            names = next;
            cap = next_cap;
        }
        names[len] = strdup(ent->d_name);
        if (!names[len]) {
            closedir(dp);
            free_string_array(names, len);
            free(dir);
            return 0;
        }
        len++;
    }
    closedir(dp);

    qsort(names, len, sizeof(*names), session_name_cmp);
    if (out_dir) {
        *out_dir = dir;
    } else {
        free(dir);
    }
    *out_names = names;
    *out_len = len;
    return 1;
}

static char *resume_session_path_by_sequence(const char *id) {
    char *dir = NULL;
    char **names = NULL;
    size_t len = 0;
    long wanted = strtol(id, NULL, 10);
    if (!load_session_names(1, 0, &names, &len, &dir)) {
        return NULL;
    }

    char *path = NULL;
    if (wanted > 0 && (size_t)wanted <= len) {
        path = path_join2(dir, names[wanted - 1]);
    } else {
        fprintf(stderr, "--resume-session: no session id %ld\n", wanted);
    }

    free_string_array(names, len);
    free(dir);
    return path;
}

static char *resume_session_path_last(void) {
    char *dir = NULL;
    char **names = NULL;
    size_t len = 0;
    if (!load_session_names(1, 0, &names, &len, &dir)) {
        return NULL;
    }
    char *path = NULL;
    if (len > 0) {
        path = path_join2(dir, names[len - 1]);
    } else {
        fprintf(stderr, "--resume-session: no recorded sessions\n");
    }

    free_string_array(names, len);
    free(dir);
    return path;
}

static int list_sessions(void) {
    char *dir = NULL;
    char **names = NULL;
    size_t len = 0;
    if (!load_session_names(0, 1, &names, &len, &dir)) {
        return 1;
    }

    if (len > 0) {
        printf("%-24s %-13s %-8s %5s %5s\n", "File", "Creation Date", "Aging", "Lines", "ID");
    }
    for (size_t i = 0; i < len; i++) {
        char *path = path_join2(dir, names[i]);
        struct stat st;
        if (path && stat(path, &st) == 0) {
            struct tm tm_time;
            localtime_r(&st.st_mtime, &tm_time);
            char date[32];
            char age[32];
            strftime(date, sizeof(date), "%Y-%m-%d", &tm_time);
            format_age(st.st_mtime, age, sizeof(age));
            long lines = count_file_lines(path);
            printf("%-24s %-13s %-8s %5ld %5zu\n", names[i], date, age, lines < 0 ? 0 : lines, i + 1);
        }
        free(path);
    }

    free_string_array(names, len);
    free(dir);
    return 0;
}

static int check_server_available(void) {
    const char *api_key = getenv("OPENAI_API_KEY");
    if (!api_key) {
        api_key = "";
    }

    const char *model = env_or_default("OPENAI_MODEL", DEFAULT_MODEL);
    const char *server = env_or_default("LLM_SERVER", DEFAULT_LLM_SERVER);
    const char *port = env_or_default("LLM_PORT", DEFAULT_LLM_PORT);
    char *url = build_url(server, port);
    if (!url) {
        fprintf(stderr, "memory allocation failed\n");
        return 1;
    }

    CURL *curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "curl initialization failed\n");
        free(url);
        return 1;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_CONNECT_ONLY, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "q/1.0");

    CURLcode rc = curl_easy_perform(curl);
    if (rc != CURLE_OK) {
        if (rc == CURLE_COULDNT_CONNECT) {
            fprintf(stderr, "LLM server is not running or not reachable: %s\n", url);
        } else if (rc == CURLE_COULDNT_RESOLVE_HOST) {
            fprintf(stderr, "LLM server host could not be resolved: %s\n", url);
        } else {
            fprintf(stderr, "LLM server preflight failed for %s: %s\n", url, curl_easy_strerror(rc));
        }
        print_server_config_hint(server, port, model, api_key);
        curl_easy_cleanup(curl);
        free(url);
        return 1;
    }

    curl_easy_cleanup(curl);
    free(url);
    return 0;
}

static int perform_query_extra(const char *input1, int think_loud, int api_logging, int color, struct code_blocks *blocks, struct chat_history *chat, char **answer_out, int quiet, const char *extra_items, int depth) {
    if (answer_out) {
        *answer_out = NULL;
    }
    request_interrupted = 0;
    request_interrupt_signal = 0;

    const char *api_key = getenv("OPENAI_API_KEY");
    if (!api_key) {
        api_key = "";
    }

    const char *model = env_or_default("OPENAI_MODEL", DEFAULT_MODEL);
    const char *server = env_or_default("LLM_SERVER", DEFAULT_LLM_SERVER);
    const char *port = env_or_default("LLM_PORT", DEFAULT_LLM_PORT);

    char *payload = build_payload_extra(model, input1, chat, extra_items);
    char *url = build_url(server, port);
    if (!payload || !url) {
        fprintf(stderr, "memory allocation failed\n");
        free(payload);
        return 1;
    }

    CURL *curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "curl initialization failed\n");
        free(payload);
        free(url);
        return 1;
    }

    struct stream_state stream = {
        .input_tokens = -1,
        .output_tokens = -1,
        .total_tokens = -1,
        .context_used_tokens = -1,
        .think_loud = think_loud,
        .turn_number = depth + 1,
        .quiet = quiet,
        .answer_color = color && isatty(STDOUT_FILENO),
        .code_color = running_on_linux_virtual_tty() ? Q_COLOR_CODE_TTY : Q_COLOR_CODE_EMULATOR
    };
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    char *auth = NULL;
    if (*api_key) {
        size_t auth_len = strlen("Authorization: Bearer ") + strlen(api_key) + 1;
        auth = malloc(auth_len);
        if (!auth) {
            fprintf(stderr, "memory allocation failed\n");
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            free(payload);
            free(url);
            return 1;
        }
        snprintf(auth, auth_len, "Authorization: Bearer %s", api_key);
        headers = curl_slist_append(headers, auth);
    }

    if (api_logging == API_LOGGING_QUERY || api_logging == API_LOGGING_BOTH) {
        log_http_request(url, model, payload, *api_key != '\0');
    }

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, stream_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, llm_timeout_seconds == -1 ? 0L : llm_timeout_seconds);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "q/1.0");
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, curl_progress_cb);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &stream);

    struct termios request_termios;
    int request_key_mode = !quiet ? begin_request_key_mode(&request_termios) : 0;

    double start = now_seconds();
    CURLcode rc = curl_easy_perform(curl);
    double elapsed = now_seconds() - start;
    end_request_key_mode(&request_termios, request_key_mode);
    if (rc == CURLE_OK) {
        flush_stream_line(&stream);
    }

    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);

    if (api_logging == API_LOGGING_RESPONSE || api_logging == API_LOGGING_BOTH) {
        clear_waiting_indicator(&stream);
        clear_thinking_indicator(&stream);
        log_http_response(status, stream.raw.data ? stream.raw.data : "");
    }

    if (rc != CURLE_OK) {
        clear_waiting_indicator(&stream);
        clear_thinking_indicator(&stream);
        if (request_interrupted) {
            if (!quiet) {
                int sig = request_interrupt_signal ? request_interrupt_signal : SIGINT;
                fprintf(stderr, "\nLLM query was interrupted via %s (%s)\n",
                        signal_name_for_message(sig), signal_keys_for_message(sig));
            }
            request_interrupted = 0;
            request_interrupt_signal = 0;
        } else if (rc == CURLE_COULDNT_CONNECT) {
            fprintf(stderr, "LLM server is not running or not reachable: %s\n", url);
            print_server_config_hint(server, port, model, api_key);
        } else if (rc == CURLE_COULDNT_RESOLVE_HOST) {
            fprintf(stderr, "LLM server host could not be resolved: %s\n", url);
            print_server_config_hint(server, port, model, api_key);
        } else {
            print_llm_request_transport_error(rc, url, server, port, model, api_key, status);
        }
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        free(auth);
        free(payload);
        free(url);
        free(stream.raw.data);
        free(stream.line.data);
        free(stream.answer_text.data);
        free(stream.tool_calls.data);
        free(stream.tool_call_id);
        free(stream.tool_name);
        free(stream.tool_arguments);
        free(stream.used_model);
        return 1;
    }

    if (status < 200 || status >= 300) {
        clear_waiting_indicator(&stream);
        clear_thinking_indicator(&stream);
        print_error_response(status, stream.raw.data ? stream.raw.data : "");
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        free(auth);
        free(payload);
        free(url);
        free(stream.raw.data);
        free(stream.line.data);
        free(stream.answer_text.data);
        free(stream.tool_calls.data);
        free(stream.tool_call_id);
        free(stream.tool_name);
        free(stream.tool_arguments);
        free(stream.used_model);
        return 1;
    }

    if (!stream.printed) {
        char *answer = get_output_text(stream.raw.data ? stream.raw.data : "");
        if (!answer) {
            if (!stream.tool_name || !stream.tool_arguments) {
                clear_waiting_indicator(&stream);
                clear_thinking_indicator(&stream);
                fprintf(stderr, "could not find streamed output text in API response\n");
                fprintf(stderr, "%s\n", stream.raw.data ? stream.raw.data : "");
                curl_slist_free_all(headers);
                curl_easy_cleanup(curl);
                free(auth);
                free(payload);
                free(url);
                free(stream.raw.data);
                free(stream.line.data);
                free(stream.answer_text.data);
                free(stream.tool_calls.data);
                free(stream.used_model);
                return 1;
            }
        } else {
            append_bytes(&stream.answer_text, answer, strlen(answer));
            if (!stream.quiet) {
                finish_llm_indicator_line(&stream);
                write_answer_text(&stream, answer);
            }
            stream.printed = answer[0] != '\0';
            free(answer);
        }
    }

    if (!stream.quiet) {
        flush_pending_backticks(&stream);
    }

    if (!stream.tool_name && stream.tool_arguments) {
        stream.tool_name = infer_tool_name_from_arguments(stream.tool_arguments);
    }

    if (stream.tool_name && stream.tool_arguments) {
        preserve_llm_indicator_line(&stream);
    } else {
        clear_waiting_indicator(&stream);
        clear_thinking_indicator(&stream);
    }
    reset_answer_color(&stream);
    if (stream.printed && !stream.quiet) {
        putchar('\n');
    }

    if (stream.tool_name && stream.tool_arguments && llm_turn_limit >= 0 && depth >= llm_turn_limit) {
        fprintf(stderr, "LLM tool follow-up turn limit reached (%ld); stopping tool follow-up loop\n", llm_turn_limit);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        free(auth);
        free(payload);
        free(url);
        free(stream.raw.data);
        free(stream.line.data);
        free(stream.answer_text.data);
        free(stream.tool_calls.data);
        free(stream.tool_call_id);
        free(stream.tool_name);
        free(stream.tool_arguments);
        free(stream.used_model);
        return 1;
    }

    if (stream.tool_name && stream.tool_arguments) {
        if (!stream.quiet) {
            print_tool_call_line(stream.tool_name, stream.tool_arguments, stream.think_loud);
        }
        char *tool_output = execute_local_tool_call(stream.tool_name, stream.tool_arguments, color);
        if (!tool_output) {
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            free(auth);
            free(payload);
            free(url);
            free(stream.raw.data);
            free(stream.line.data);
            free(stream.answer_text.data);
            free(stream.tool_calls.data);
            free(stream.tool_call_id);
            free(stream.tool_name);
            free(stream.tool_arguments);
            free(stream.used_model);
            return 1;
        }
        char *tool_items = build_tool_result_items(stream.tool_call_id, stream.tool_name, stream.tool_arguments, tool_output);
        free(tool_output);
        int tool_rc = 1;
        if (tool_items) {
            tool_rc = perform_query_extra(input1, stream.think_loud, api_logging, color, blocks, chat, answer_out, quiet, tool_items, depth + 1);
        } else {
            fprintf(stderr, "tool call handling failed: memory allocation failed\n");
        }
        free(tool_items);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        free(auth);
        free(payload);
        free(url);
        free(stream.raw.data);
        free(stream.line.data);
        free(stream.answer_text.data);
        free(stream.tool_calls.data);
        free(stream.tool_call_id);
        free(stream.tool_name);
        free(stream.tool_arguments);
        free(stream.used_model);
        return tool_rc;
    }

    long p_rate = elapsed > 0.0 && stream.input_tokens >= 0 ? (long)((double)stream.input_tokens / elapsed + 0.5) : 0;
    long t_rate = elapsed > 0.0 && stream.total_tokens >= 0 ? (long)((double)stream.total_tokens / elapsed + 0.5) : 0;
    long duration = elapsed >= 0.0 ? (long)(elapsed + 0.5) : 0;
    if (!stream.quiet) {
        printf("\n[%s, p: %ld, t: %ld, d: %ld",
               stream.used_model ? stream.used_model : model, p_rate, t_rate, duration);
        if (stream.input_tokens >= 0 || stream.output_tokens >= 0 || stream.total_tokens >= 0) {
            printf(", tio:%ld+%ld=%ld",
                   stream.input_tokens, stream.output_tokens, stream.total_tokens);
        }
        printf("]\n");
    }

    if (blocks) {
        code_blocks_clear(blocks);
        extract_shell_code_blocks(stream.answer_text.data ? stream.answer_text.data : "", blocks);
        print_numbered_code_blocks(blocks, color);
    }

    if (chat && chat->enabled) {
        if (!chat_history_add(chat, "user", input1) ||
            !chat_history_add(chat, "assistant", stream.answer_text.data ? stream.answer_text.data : "")) {
            fprintf(stderr, "context history update failed: memory allocation failed\n");
        } else if (!chat_history_append_file(chat, "user", input1) ||
                   !chat_history_append_file(chat, "assistant", stream.answer_text.data ? stream.answer_text.data : "")) {
            fprintf(stderr, "context history file update failed\n");
        }
    }

    if (answer_out) {
        *answer_out = strdup(stream.answer_text.data ? stream.answer_text.data : "");
        if (!*answer_out) {
            fprintf(stderr, "completion cache update failed: memory allocation failed\n");
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    free(auth);
    free(payload);
    free(url);
    free(stream.raw.data);
    free(stream.line.data);
    free(stream.answer_text.data);
    free(stream.tool_calls.data);
    free(stream.tool_call_id);
    free(stream.tool_name);
    free(stream.tool_arguments);
    free(stream.used_model);
    return 0;
}

static int perform_query(const char *input1, int think_loud, int api_logging, int color, struct code_blocks *blocks, struct chat_history *chat, char **answer_out, int quiet) {
    return perform_query_extra(input1, think_loud, api_logging, color, blocks, chat, answer_out, quiet, NULL, 0);
}

static int is_safe_command_word(const char *word) {
    if (!word || !*word) {
        return 0;
    }
    for (const unsigned char *p = (const unsigned char *)word; *p; p++) {
        if (!isalnum(*p) && *p != '_' && *p != '-' && *p != '.' && *p != '+' && *p != '/') {
            return 0;
        }
    }
    return 1;
}

static char *shell_quote_word(const char *s) {
    size_t len = 2;
    for (const char *p = s; *p; p++) {
        len += *p == '\'' ? 4 : 1;
    }

    char *out = malloc(len + 1);
    if (!out) {
        return NULL;
    }

    char *q = out;
    *q++ = '\'';
    for (const char *p = s; *p; p++) {
        if (*p == '\'') {
            memcpy(q, "'\\''", 4);
            q += 4;
        } else {
            *q++ = *p;
        }
    }
    *q++ = '\'';
    *q = '\0';
    return out;
}

static char *capture_command_limited(const char *cmd, size_t limit) {
    FILE *fp = popen(cmd, "r");
    if (!fp) {
        return strdup("");
    }

    struct buffer out = {0};
    char chunk[2048];
    while (fgets(chunk, sizeof(chunk), fp)) {
        size_t n = strlen(chunk);
        if (out.len + n > limit) {
            n = limit > out.len ? limit - out.len : 0;
        }
        if (n > 0 && !append_bytes(&out, chunk, n)) {
            free(out.data);
            pclose(fp);
            return strdup("");
        }
        if (out.len >= limit) {
            break;
        }
    }
    pclose(fp);

    if (!out.data) {
        return strdup("");
    }
    return out.data;
}

static char *command_before_cursor_token(const struct buffer *buf, size_t cursor) {
    if (!buf->data || cursor > buf->len) {
        return NULL;
    }

    size_t token_start = cursor;
    while (token_start > 0 && !isspace((unsigned char)buf->data[token_start - 1])) {
        token_start--;
    }
    if (token_start == 0) {
        return NULL;
    }

    size_t seg_start = token_start;
    while (seg_start > 0) {
        char c = buf->data[seg_start - 1];
        if (c == '|' || c == ';' || c == '&' || c == '<' || c == '>' || c == '\n') {
            break;
        }
        seg_start--;
    }

    size_t len = token_start - seg_start;
    char *segment = malloc(len + 1);
    if (!segment) {
        return NULL;
    }
    memcpy(segment, buf->data + seg_start, len);
    segment[len] = '\0';
    char *cmd = first_shell_word(segment);
    free(segment);
    return cmd;
}

static int has_no_manual_entry(const char *text, const char *cmd) {
    (void)cmd;
    return text && contains_word_ci(text, "no manual entry");
}

static char *load_or_build_command_options(const char *cmd) {
    if (!is_safe_command_word(cmd)) {
        return NULL;
    }

    char *cache_path = completion_cache_path(cmd, 0);
    if (cache_path) {
        char *cached = read_text_file(cache_path);
        if (cached) {
            free(cache_path);
            return cached;
        }
        free(cache_path);
    }

    char *quoted = shell_quote_word(cmd);
    if (!quoted) {
        return NULL;
    }

    size_t man_cmd_len = strlen(quoted) + 64;
    char *man_cmd = malloc(man_cmd_len);
    size_t help_cmd_len = strlen(quoted) + 32;
    char *help_cmd = malloc(help_cmd_len);
    if (!man_cmd || !help_cmd) {
        free(quoted);
        free(man_cmd);
        free(help_cmd);
        return NULL;
    }

    snprintf(man_cmd, man_cmd_len, "MANWIDTH=100 man %s 2>&1 | col -b", quoted);
    snprintf(help_cmd, help_cmd_len, "%s --help 2>&1", quoted);
    free(quoted);

    char *man_text = capture_command_limited(man_cmd, 24000);
    char *source_text = man_text;
    const char *source_label = "MAN PAGE";
    if (man_text && has_no_manual_entry(man_text, cmd)) {
        free(man_text);
        source_text = capture_command_limited(help_cmd, 16000);
        source_label = "--help OUTPUT";
    }
    free(man_cmd);
    free(help_cmd);
    if (!source_text) {
        return NULL;
    }

    const char *fmt =
        "Command: %s\n"
        "Current option text: %s\n\n"
        "Using the command documentation below, summarize all available command-line arguments for this command. "
        "Return plain text only: no Markdown, no backticks, no code formatting, no asterisk bullets. "
        "Use one option per line in this exact format: option: description. Align all colons in the same column by padding spaces after the option. "
        "Do not narrow the result; this output will be cached and filtered locally later.\n\n"
        "%s:\n%s\n";
    size_t prompt_len = snprintf(NULL, 0, fmt, cmd, "", source_label, source_text);
    char *prompt = malloc(prompt_len + 1);
    if (!prompt) {
        free(source_text);
        return NULL;
    }
    snprintf(prompt, prompt_len + 1, fmt, cmd, "", source_label, source_text);
    free(source_text);

    char *answer = NULL;
    int was_raw = terminal_raw_active;
    if (was_raw) {
        restore_terminal_if_needed();
    }
    if (perform_query(prompt, 0, 0, 0, NULL, NULL, &answer, 2) == 0 && answer && *answer) {
        cache_path = completion_cache_path(cmd, 1);
        if (cache_path && !write_text_file(cache_path, answer)) {
            fprintf(stderr, "completion: failed to write cache %s\n", cache_path);
        }
        free(cache_path);
    }
    if (was_raw) {
        enable_raw_from_saved_if_needed();
    }
    free(prompt);
    return answer;
}

struct history {
    char **items;
    size_t len;
    size_t cap;
};

static void history_free(struct history *hist) {
    for (size_t i = 0; i < hist->len; i++) {
        free(hist->items[i]);
    }
    free(hist->items);
}

static int history_add(struct history *hist, const char *line) {
    if (!line || !*line) {
        return 1;
    }
    if (hist->len > 0 && strcmp(hist->items[hist->len - 1], line) == 0) {
        return 1;
    }
    if (hist->len == hist->cap) {
        size_t next_cap = hist->cap ? hist->cap * 2 : 32;
        char **next = realloc(hist->items, next_cap * sizeof(*next));
        if (!next) {
            return 0;
        }
        hist->items = next;
        hist->cap = next_cap;
    }
    hist->items[hist->len] = strdup(line);
    if (!hist->items[hist->len]) {
        return 0;
    }
    hist->len++;
    return 1;
}

static int history_load_from_chat(struct history *hist, const struct chat_history *chat) {
    if (!chat || !chat->enabled) {
        return 1;
    }
    if (chat->input_len > 0) {
        for (size_t i = 0; i < chat->input_len; i++) {
            if (!history_add(hist, chat->input_items[i])) {
                return 0;
            }
        }
        return 1;
    }
    for (size_t i = 0; i < chat->len; i++) {
        if (strcmp(chat->items[i].role, "user") == 0 && !history_add(hist, chat->items[i].content)) {
            return 0;
        }
    }
    return 1;
}

static void record_input_history(struct history *hist, struct chat_history *chat, const char *line) {
    if (!history_add(hist, line)) {
        fprintf(stderr, "history: memory allocation failed\n");
    } else if (!chat_history_record_input(chat, line)) {
        fprintf(stderr, "session history update failed\n");
    }
}

static int history_add_block_lines(struct history *hist, const char *block) {
    const char *p = block ? block : "";
    int added = 0;

    while (*p) {
        const char *end = strchr(p, '\n');
        size_t len = end ? (size_t)(end - p) : strlen(p);
        while (len > 0 && isspace((unsigned char)*p)) {
            p++;
            len--;
        }
        while (len > 0 && isspace((unsigned char)p[len - 1])) {
            len--;
        }
        if (len > 0) {
            char *line = malloc(len + 1);
            if (!line) {
                return 0;
            }
            memcpy(line, p, len);
            line[len] = '\0';
            int ok = history_add(hist, line);
            free(line);
            if (!ok) {
                return 0;
            }
            added = 1;
        }
        if (!end) {
            break;
        }
        p = end + 1;
    }

    return added || history_add(hist, block);
}

static int chat_history_record_block_lines(struct chat_history *chat, const char *block) {
    const char *p = block ? block : "";
    int recorded = 0;

    while (*p) {
        const char *end = strchr(p, '\n');
        size_t len = end ? (size_t)(end - p) : strlen(p);
        while (len > 0 && isspace((unsigned char)*p)) {
            p++;
            len--;
        }
        while (len > 0 && isspace((unsigned char)p[len - 1])) {
            len--;
        }
        if (len > 0) {
            char *line = malloc(len + 1);
            if (!line) {
                return 0;
            }
            memcpy(line, p, len);
            line[len] = '\0';
            int ok = chat_history_record_input(chat, line);
            free(line);
            if (!ok) {
                return 0;
            }
            recorded = 1;
        }
        if (!end) {
            break;
        }
        p = end + 1;
    }

    return recorded || chat_history_record_input(chat, block);
}

static int append_char(struct buffer *buf, char c) {
    return append_bytes(buf, &c, 1);
}

static int set_buffer_string(struct buffer *buf, const char *s) {
    size_t len = strlen(s);
    char *next = realloc(buf->data, len + 1);
    if (!next) {
        return 0;
    }
    buf->data = next;
    memcpy(buf->data, s, len + 1);
    buf->len = len;
    return 1;
}

static int insert_char_at(struct buffer *buf, size_t pos, char c) {
    if (pos > buf->len) {
        pos = buf->len;
    }
    char *next = realloc(buf->data, buf->len + 2);
    if (!next) {
        return 0;
    }
    buf->data = next;
    if (buf->len == 0) {
        buf->data[0] = '\0';
    }
    memmove(buf->data + pos + 1, buf->data + pos, buf->len - pos + 1);
    buf->data[pos] = c;
    buf->len++;
    return 1;
}

static int insert_text_at(struct buffer *buf, size_t *pos, const char *text) {
    if (!text || !*text) {
        return 1;
    }
    if (*pos > buf->len) {
        *pos = buf->len;
    }
    size_t text_len = strlen(text);
    char *next = realloc(buf->data, buf->len + text_len + 1);
    if (!next) {
        return 0;
    }
    buf->data = next;
    if (buf->len == 0) {
        buf->data[0] = '\0';
    }
    memmove(buf->data + *pos + text_len, buf->data + *pos, buf->len - *pos + 1);
    memcpy(buf->data + *pos, text, text_len);
    buf->len += text_len;
    *pos += text_len;
    return 1;
}

static char *buffer_slice(const struct buffer *buf, size_t start, size_t end) {
    if (!buf || !buf->data || start >= end || end > buf->len) {
        return NULL;
    }
    char *out = malloc(end - start + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, buf->data + start, end - start);
    out[end - start] = '\0';
    return out;
}

static void clipboard_write_text(const char *text) {
    if (!text || !*text) {
        return;
    }
    const char *cmds[] = {
        "wl-copy 2>/dev/null",
        "xclip -selection clipboard 2>/dev/null",
        "xsel --clipboard --input 2>/dev/null",
        NULL
    };
    for (size_t i = 0; cmds[i]; i++) {
        FILE *fp = popen(cmds[i], "w");
        if (!fp) {
            continue;
        }
        fwrite(text, 1, strlen(text), fp);
        int rc = pclose(fp);
        if (rc == 0) {
            return;
        }
    }
}

static char *clipboard_read_text(void) {
    const char *cmds[] = {
        "wl-paste --no-newline 2>/dev/null",
        "xclip -selection clipboard -o 2>/dev/null",
        "xsel --clipboard --output 2>/dev/null",
        NULL
    };
    for (size_t i = 0; cmds[i]; i++) {
        FILE *fp = popen(cmds[i], "r");
        if (!fp) {
            continue;
        }
        struct buffer out = {0};
        char chunk[1024];
        size_t nread;
        while ((nread = fread(chunk, 1, sizeof(chunk), fp)) > 0) {
            if (!append_bytes(&out, chunk, nread)) {
                free(out.data);
                pclose(fp);
                return NULL;
            }
        }
        int rc = pclose(fp);
        if (rc == 0 && out.data) {
            return out.data;
        }
        free(out.data);
    }
    return NULL;
}

static void delete_range(struct buffer *buf, size_t start, size_t end) {
    if (start >= end || end > buf->len) {
        return;
    }
    memmove(buf->data + start, buf->data + end, buf->len - end + 1);
    buf->len -= end - start;
}

static size_t move_word_forward_pos(const struct buffer *buf, size_t cursor) {
    const char *data = buf->data ? buf->data : "";
    while (cursor < buf->len && isspace((unsigned char)data[cursor])) {
        cursor++;
    }
    while (cursor < buf->len && !isspace((unsigned char)data[cursor])) {
        cursor++;
    }
    return cursor;
}

static size_t move_word_backward_pos(const struct buffer *buf, size_t cursor) {
    const char *data = buf->data ? buf->data : "";
    while (cursor > 0 && isspace((unsigned char)data[cursor - 1])) {
        cursor--;
    }
    while (cursor > 0 && !isspace((unsigned char)data[cursor - 1])) {
        cursor--;
    }
    return cursor;
}

static int prompt_visible_cols(int line_no) {
    int digits = 1;
    for (int n = line_no; n >= 10; n /= 10) {
        digits++;
    }
    return digits + 1 + 1 + 8;
}

static void print_repl_prompt(int line_no, int color) {
    if (color) {
        fprintf(stderr, Q_COLOR_PROMPT_LINE_NO "%d" Q_COLOR_RESET " $        ", line_no);
    } else {
        fprintf(stderr, "%d $        ", line_no);
    }
}

static int terminal_width(void);

static void editor_position_for_index(const char *data, size_t len, size_t index, size_t prompt_cols, int width, int *row, int *col) {
    int r = 0;
    int c = (int)prompt_cols;
    if (width <= 0) {
        width = 80;
    }

    for (size_t i = 0; i < index && i < len; i++) {
        if (data[i] == '\n') {
            r++;
            c = 0;
            continue;
        }
        if (c >= width) {
            r++;
            c = 0;
        }
        c++;
    }

    *row = r;
    *col = c;
}

static void write_editor_text_wrapped(const char *data, size_t len, size_t prompt_cols, int width) {
    int c = (int)prompt_cols;
    if (width <= 0) {
        width = 80;
    }
    for (size_t i = 0; i < len; i++) {
        if (data[i] == '\n') {
            fputc('\n', stderr);
            c = 0;
            continue;
        }
        if (c >= width) {
            fputc('\n', stderr);
            c = 0;
        }
        fputc(data[i], stderr);
        c++;
    }
}

static void refresh_editor_line(int line_no, const struct buffer *buf, size_t cursor, int color, int *rendered_rows, int *rendered_cursor_row) {
    size_t len = buf->data ? buf->len : 0;
    const char *data = buf->data ? buf->data : "";
    int width = terminal_width();
    if (width > 1) {
        width--;
    }
    size_t prompt_cols = (size_t)prompt_visible_cols(line_no);
    int old_cursor_row = rendered_cursor_row && *rendered_cursor_row > 0 ? *rendered_cursor_row : 0;

    fputc('\r', stderr);
    if (old_cursor_row > 0) {
        fprintf(stderr, "\033[%dA", old_cursor_row);
    }
    fputs("\033[J", stderr);
    print_repl_prompt(line_no, color);
    write_editor_text_wrapped(data, len, prompt_cols, width);
    fputs("\033[K", stderr);

    int end_row = 0;
    int end_col = 0;
    int cursor_row = 0;
    int cursor_col = 0;
    editor_position_for_index(data, len, len, prompt_cols, width, &end_row, &end_col);
    editor_position_for_index(data, len, cursor, prompt_cols, width, &cursor_row, &cursor_col);

    if (end_row > cursor_row) {
        fprintf(stderr, "\033[%dA", end_row - cursor_row);
    }
    if (end_col > 0) {
        fprintf(stderr, "\033[%dD", end_col);
    }
    if (cursor_col > 0) {
        fprintf(stderr, "\033[%dC", cursor_col);
    }

    if (rendered_rows) {
        *rendered_rows = end_row + 1;
    }
    if (rendered_cursor_row) {
        *rendered_cursor_row = cursor_row;
    }
    fflush(stderr);
}

static int enable_raw_mode(struct termios *orig) {
    if (tcgetattr(STDIN_FILENO, orig) != 0) {
        return 0;
    }
    saved_termios = *orig;
    saved_termios_valid = 1;
    struct termios raw = *orig;
    raw.c_lflag &= (tcflag_t)~(ECHO | ICANON | IEXTEN);
    raw.c_iflag &= (tcflag_t)~(IXON | ICRNL);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) != 0) {
        return 0;
    }
    terminal_raw_active = 1;
    return 1;
}

static void restore_termios(const struct termios *orig) {
    if (orig) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, orig);
    } else {
        restore_terminal_if_needed();
        return;
    }
    terminal_raw_active = 0;
}

static int add_completion_match(char ***items, size_t *len, size_t *cap, const char *name, int is_dir) {
    if (*len == *cap) {
        size_t next_cap = *cap ? *cap * 2 : 32;
        char **next = realloc(*items, next_cap * sizeof(*next));
        if (!next) {
            return 0;
        }
        *items = next;
        *cap = next_cap;
    }

    size_t n = strlen(name);
    char *item = malloc(n + (is_dir ? 2 : 1));
    if (!item) {
        return 0;
    }
    memcpy(item, name, n);
    if (is_dir) {
        item[n++] = '/';
    }
    item[n] = '\0';
    (*items)[(*len)++] = item;
    return 1;
}

static int add_plain_completion_match(char ***items, size_t *len, size_t *cap, const char *name) {
    if (*len == *cap) {
        size_t next_cap = *cap ? *cap * 2 : 32;
        char **next = realloc(*items, next_cap * sizeof(*next));
        if (!next) {
            return 0;
        }
        *items = next;
        *cap = next_cap;
    }
    (*items)[*len] = strdup(name);
    if (!(*items)[*len]) {
        return 0;
    }
    (*len)++;
    return 1;
}

static int completion_path_is_dir(const char *path, int d_type) {
    if (d_type == DT_DIR) {
        return 1;
    }
    if (d_type != DT_UNKNOWN) {
        return 0;
    }
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int completion_path_is_executable_file(const char *path, int d_type) {
    if (d_type == DT_DIR) {
        return 0;
    }
    if (d_type != DT_REG && d_type != DT_UNKNOWN) {
        return 0;
    }
    struct stat st;
    if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) {
        return 0;
    }
    return access(path, X_OK) == 0;
}

static int completion_path_is_regular_file(const char *path, int d_type) {
    if (d_type == DT_REG) {
        return 1;
    }
    if (d_type != DT_UNKNOWN && d_type != DT_LNK) {
        return 0;
    }
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static char *completion_expand_tilde_path(const char *path) {
    if (!path || path[0] != '~' || (path[1] != '/' && path[1] != '\0')) {
        return path ? strdup(path) : NULL;
    }
    const char *home = getenv("HOME");
    if (!home || !*home) {
        return strdup(path);
    }
    size_t home_len = strlen(home);
    size_t rest_off = path[1] == '/' ? 2 : 1;
    const char *rest = path + rest_off;
    size_t len = home_len + 1 + strlen(rest) + 1;
    char *out = malloc(len);
    if (!out) {
        return NULL;
    }
    if (*rest) {
        snprintf(out, len, "%s/%s", home, rest);
    } else {
        snprintf(out, len, "%s", home);
    }
    return out;
}

static size_t completion_insert_len(const char *item) {
    const char *tab = strchr(item, '\t');
    return tab ? (size_t)(tab - item) : strlen(item);
}

static const char *completion_display_text(const char *item) {
    const char *tab = strchr(item, '\t');
    return tab ? tab + 1 : item;
}

static char *common_prefix(char **items, size_t len, const char *fallback) {
    if (len == 0) {
        return strdup(fallback ? fallback : "");
    }

    size_t prefix_len = completion_insert_len(items[0]);
    for (size_t i = 1; i < len; i++) {
        size_t item_len = completion_insert_len(items[i]);
        if (item_len < prefix_len) {
            prefix_len = item_len;
        }
        size_t j = 0;
        while (j < prefix_len && items[0][j] == items[i][j]) {
            j++;
        }
        prefix_len = j;
    }

    char *out = malloc(prefix_len + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, items[0], prefix_len);
    out[prefix_len] = '\0';
    return out;
}

static int match_exists(char **items, size_t len, const char *name) {
    for (size_t i = 0; i < len; i++) {
        size_t insert_len = completion_insert_len(items[i]);
        if (strlen(name) == insert_len && strncmp(items[i], name, insert_len) == 0) {
            return 1;
        }
    }
    return 0;
}

static int starts_with_ci(const char *s, const char *prefix) {
    if (!prefix || !*prefix) {
        return 1;
    }
    if (!s) {
        return 0;
    }
    while (*prefix) {
        if (tolower((unsigned char)*s) != tolower((unsigned char)*prefix)) {
            return 0;
        }
        s++;
        prefix++;
    }
    return 1;
}

static int add_option_completion_match(char ***items, size_t *len, size_t *cap, const char *insert, const char *field, size_t field_len, const char *desc, size_t desc_len) {
    if (!insert || !*insert || match_exists(*items, *len, insert)) {
        return 1;
    }
    if (*len == *cap) {
        size_t next_cap = *cap ? *cap * 2 : 32;
        char **next = realloc(*items, next_cap * sizeof(*next));
        if (!next) {
            return 0;
        }
        *items = next;
        *cap = next_cap;
    }

    while (field_len > 0 && isspace((unsigned char)field[0])) {
        field++;
        field_len--;
    }
    while (field_len > 0 && isspace((unsigned char)field[field_len - 1])) {
        field_len--;
    }
    while (desc_len > 0 && desc && isspace((unsigned char)desc[0])) {
        desc++;
        desc_len--;
    }

    size_t insert_len = strlen(insert);
    size_t pad = field_len < 15 ? 15 - field_len : 1;
    size_t total = insert_len + 1 + field_len + pad + 3 + desc_len + 1;
    char *item = malloc(total);
    if (!item) {
        return 0;
    }
    char *p = item;
    memcpy(p, insert, insert_len);
    p += insert_len;
    *p++ = '\t';
    memcpy(p, field, field_len);
    p += field_len;
    memset(p, ' ', pad);
    p += pad;
    memcpy(p, "-- ", 3);
    p += 3;
    if (desc && desc_len > 0) {
        memcpy(p, desc, desc_len);
        p += desc_len;
    }
    *p = '\0';
    (*items)[(*len)++] = item;
    return 1;
}

static int option_matches_prefix(const char *option, const char *partial) {
    if (!partial || !*partial || strcmp(partial, "-") == 0) {
        return option && option[0] == '-';
    }
    return strncmp(option, partial, strlen(partial)) == 0;
}

static int add_option_tokens_from_field(char ***items, size_t *len, size_t *cap, const char *field, size_t field_len, const char *desc, size_t desc_len, const char *partial) {
    size_t i = 0;
    while (i < field_len) {
        while (i < field_len && field[i] != '-') {
            i++;
        }
        if (i >= field_len) {
            break;
        }

        size_t start = i;
        while (i < field_len &&
               !isspace((unsigned char)field[i]) &&
               field[i] != ',' &&
               field[i] != ';' &&
               field[i] != '[' &&
               field[i] != '=' &&
               field[i] != ':') {
            i++;
        }

        size_t tok_len = i - start;
        while (tok_len > 0 && (field[start + tok_len - 1] == ',' || field[start + tok_len - 1] == '.')) {
            tok_len--;
        }
        if (tok_len == 0) {
            continue;
        }

        char *tok = malloc(tok_len + 1);
        if (!tok) {
            return 0;
        }
        memcpy(tok, field + start, tok_len);
        tok[tok_len] = '\0';
        if (option_matches_prefix(tok, partial) && !add_option_completion_match(items, len, cap, tok, field, field_len, desc, desc_len)) {
            free(tok);
            return 0;
        }
        free(tok);
    }
    return 1;
}

static char **option_completion_matches_from_summary(const char *summary, const char *partial) {
    char **items = NULL;
    size_t len = 0;
    size_t cap = 0;
    const char *p = summary ? summary : "";

    while (*p) {
        const char *end = strchr(p, '\n');
        size_t line_len = end ? (size_t)(end - p) : strlen(p);
        const char *colon = memchr(p, ':', line_len);
        size_t field_len = colon ? (size_t)(colon - p) : line_len;
        const char *desc = colon ? colon + 1 : "";
        size_t desc_len = colon ? line_len - (size_t)(desc - p) : 0;
        while (field_len > 0 && isspace((unsigned char)p[field_len - 1])) {
            field_len--;
        }
        if (!add_option_tokens_from_field(&items, &len, &cap, p, field_len, desc, desc_len, partial)) {
            for (size_t i = 0; i < len; i++) {
                free(items[i]);
            }
            free(items);
            return NULL;
        }
        if (!end) {
            break;
        }
        p = end + 1;
    }

    if (len == 0) {
        free(items);
        return NULL;
    }

    char **matches = calloc(len + 2, sizeof(*matches));
    if (!matches) {
        for (size_t i = 0; i < len; i++) {
            free(items[i]);
        }
        free(items);
        return NULL;
    }
    matches[0] = common_prefix(items, len, partial ? partial : "");
    for (size_t i = 0; i < len; i++) {
        matches[i + 1] = items[i];
    }
    free(items);
    return matches;
}

static char **cwd_completion_matches(const char *prefix, int include_regular_files) {
    DIR *dp = opendir(".");
    if (!dp) {
        perror("opendir");
        return NULL;
    }

    char **items = NULL;
    size_t len = 0;
    size_t cap = 0;
    size_t prefix_len = prefix ? strlen(prefix) : 0;
    struct dirent *ent;
    while ((ent = readdir(dp)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }
        if (prefix_len > 0 && !starts_with_ci(ent->d_name, prefix)) {
            continue;
        }
        int is_dir = completion_path_is_dir(ent->d_name, ent->d_type);
        int is_regular_file = !is_dir && include_regular_files &&
                              completion_path_is_regular_file(ent->d_name, ent->d_type);
        if (!is_dir && !is_regular_file) {
            continue;
        }
        if (!add_completion_match(&items, &len, &cap, ent->d_name, is_dir)) {
            closedir(dp);
            for (size_t i = 0; i < len; i++) {
                free(items[i]);
            }
            free(items);
            return NULL;
        }
    }
    closedir(dp);

    if (len == 0) {
        free(items);
        return NULL;
    }

    char **matches = calloc(len + 2, sizeof(*matches));
    if (!matches) {
        for (size_t i = 0; i < len; i++) {
            free(items[i]);
        }
        free(items);
        return NULL;
    }
    matches[0] = common_prefix(items, len, prefix ? prefix : "");
    for (size_t i = 0; i < len; i++) {
        matches[i + 1] = items[i];
    }
    free(items);
    return matches;
}

static char **path_completion_matches(const char *prefix, int include_regular_files) {
    const char *slash = strrchr(prefix, '/');
    if (!slash) {
        return cwd_completion_matches(prefix, include_regular_files);
    }

    size_t dir_len = (size_t)(slash - prefix);
    const char *base = slash + 1;
    size_t base_len = strlen(base);
    char *dir = NULL;
    char *real_dir = NULL;
    char *display_dir = NULL;

    if (dir_len == 0) {
        dir = strdup("/");
        display_dir = strdup("/");
    } else {
        dir = malloc(dir_len + 1);
        display_dir = malloc(dir_len + 2);
        if (dir) {
            memcpy(dir, prefix, dir_len);
            dir[dir_len] = '\0';
        }
        if (display_dir) {
            memcpy(display_dir, prefix, dir_len);
            display_dir[dir_len] = '/';
            display_dir[dir_len + 1] = '\0';
        }
    }
    if (!dir || !display_dir) {
        free(dir);
        free(display_dir);
        return NULL;
    }

    real_dir = completion_expand_tilde_path(dir);
    if (!real_dir) {
        free(dir);
        free(display_dir);
        return NULL;
    }

    int allow_executable_files = include_regular_files || strncmp(prefix, "./", 2) == 0;
    DIR *dp = opendir(real_dir);
    if (!dp) {
        free(dir);
        free(real_dir);
        free(display_dir);
        return NULL;
    }

    char **items = NULL;
    size_t len = 0;
    size_t cap = 0;
    struct dirent *ent;
    while ((ent = readdir(dp)) != NULL) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
            continue;
        }
        if (base_len > 0 && !starts_with_ci(ent->d_name, base)) {
            continue;
        }

        size_t name_len = strlen(display_dir) + strlen(ent->d_name) + 1;
        char *name = malloc(name_len);
        if (!name) {
            continue;
        }
        snprintf(name, name_len, "%s%s", display_dir, ent->d_name);
        size_t real_name_len = strlen(real_dir) + 1 + strlen(ent->d_name) + 1;
        char *real_name = malloc(real_name_len);
        if (!real_name) {
            free(name);
            continue;
        }
        snprintf(real_name, real_name_len, "%s/%s", real_dir, ent->d_name);
        int is_dir = completion_path_is_dir(real_name, ent->d_type);
        int is_regular_file = !is_dir && include_regular_files &&
                              completion_path_is_regular_file(real_name, ent->d_type);
        int is_exec_file = !is_dir && allow_executable_files && completion_path_is_executable_file(real_name, ent->d_type);
        free(real_name);
        if (!is_dir && !is_regular_file && !is_exec_file) {
            free(name);
            continue;
        }
        int ok = add_completion_match(&items, &len, &cap, name, is_dir);
        free(name);
        if (!ok) {
            closedir(dp);
            for (size_t i = 0; i < len; i++) {
                free(items[i]);
            }
            free(items);
            free(dir);
            free(real_dir);
            free(display_dir);
            return NULL;
        }
    }
    closedir(dp);
    free(dir);
    free(real_dir);
    free(display_dir);

    if (len == 0) {
        free(items);
        return NULL;
    }

    char **matches = calloc(len + 2, sizeof(*matches));
    if (!matches) {
        for (size_t i = 0; i < len; i++) {
            free(items[i]);
        }
        free(items);
        return NULL;
    }
    matches[0] = common_prefix(items, len, prefix);
    for (size_t i = 0; i < len; i++) {
        matches[i + 1] = items[i];
    }
    free(items);
    return matches;
}

static char **slash_command_completion_matches(const char *prefix) {
    static const char *commands[] = {
        "/help",
        "/keys",
        "/show-system-prompt",
        "/set-system-prompt",
        "/exit",
        "/llm-timeout",
        "/llm-turn-limit",
        "/think-loud",
        "/api-logging",
        "/add-mcp-server",
        "/remove-mcp-server",
        "/list-mcp-servers",
        "/note",
        "/truncate-context",
        "/clear-completion-cache",
        NULL
    };

    char **items = NULL;
    size_t len = 0;
    size_t cap = 0;
    size_t prefix_len = prefix ? strlen(prefix) : 0;
    for (size_t i = 0; commands[i]; i++) {
        if (prefix_len > 0 && strncmp(commands[i], prefix, prefix_len) != 0) {
            continue;
        }
        if (!add_plain_completion_match(&items, &len, &cap, commands[i])) {
            for (size_t j = 0; j < len; j++) {
                free(items[j]);
            }
            free(items);
            return NULL;
        }
    }

    if (len == 0) {
        free(items);
        return NULL;
    }

    char **matches = calloc(len + 2, sizeof(*matches));
    if (!matches) {
        for (size_t i = 0; i < len; i++) {
            free(items[i]);
        }
        free(items);
        return NULL;
    }
    matches[0] = common_prefix(items, len, prefix ? prefix : "");
    for (size_t i = 0; i < len; i++) {
        matches[i + 1] = items[i];
    }
    free(items);
    return matches;
}

static char **command_completion_matches(const char *prefix, const struct shell_aliases *aliases) {
    char **items = NULL;
    size_t len = 0;
    size_t cap = 0;
    size_t prefix_len = prefix ? strlen(prefix) : 0;

    if (aliases) {
        for (size_t i = 0; i < aliases->len; i++) {
            if (prefix_len > 0 && strncmp(aliases->names[i], prefix, prefix_len) != 0) {
                continue;
            }
            if (!match_exists(items, len, aliases->names[i]) &&
                !add_plain_completion_match(&items, &len, &cap, aliases->names[i])) {
                for (size_t j = 0; j < len; j++) {
                    free(items[j]);
                }
                free(items);
                return NULL;
            }
        }
    }

    DIR *dp = opendir(".");
    if (dp) {
        struct dirent *ent;
        while ((ent = readdir(dp)) != NULL) {
            if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
                continue;
            }
            if (prefix_len > 0 && !starts_with_ci(ent->d_name, prefix)) {
                continue;
            }
            int is_dir = completion_path_is_dir(ent->d_name, ent->d_type);
            if (!is_dir || match_exists(items, len, ent->d_name)) {
                continue;
            }
            if (!add_completion_match(&items, &len, &cap, ent->d_name, is_dir)) {
                closedir(dp);
                for (size_t i = 0; i < len; i++) {
                    free(items[i]);
                }
                free(items);
                return NULL;
            }
        }
        closedir(dp);
    }

    if (len == 0) {
        free(items);
        return NULL;
    }

    char **matches = calloc(len + 2, sizeof(*matches));
    if (!matches) {
        for (size_t i = 0; i < len; i++) {
            free(items[i]);
        }
        free(items);
        return NULL;
    }
    matches[0] = common_prefix(items, len, prefix ? prefix : "");
    for (size_t i = 0; i < len; i++) {
        matches[i + 1] = items[i];
    }
    free(items);
    return matches;
}

static char **env_completion_matches(const char *prefix) {
    char **items = NULL;
    size_t len = 0;
    size_t cap = 0;
    size_t prefix_len = prefix ? strlen(prefix) : 0;

    for (char **env = environ; env && *env; env++) {
        const char *eq = strchr(*env, '=');
        if (!eq || eq == *env) {
            continue;
        }
        size_t name_len = (size_t)(eq - *env);
        size_t candidate_len = name_len + 2;
        char *candidate = malloc(candidate_len);
        if (!candidate) {
            for (size_t i = 0; i < len; i++) {
                free(items[i]);
            }
            free(items);
            return NULL;
        }
        candidate[0] = '$';
        memcpy(candidate + 1, *env, name_len);
        candidate[name_len + 1] = '\0';
        if (prefix_len == 0 || strncmp(candidate, prefix, prefix_len) == 0) {
            if (!add_plain_completion_match(&items, &len, &cap, candidate)) {
                free(candidate);
                for (size_t i = 0; i < len; i++) {
                    free(items[i]);
                }
                free(items);
                return NULL;
            }
        }
        free(candidate);
    }

    if (len == 0) {
        free(items);
        return NULL;
    }

    char **matches = calloc(len + 2, sizeof(*matches));
    if (!matches) {
        for (size_t i = 0; i < len; i++) {
            free(items[i]);
        }
        free(items);
        return NULL;
    }
    matches[0] = common_prefix(items, len, prefix ? prefix : "$");
    for (size_t i = 0; i < len; i++) {
        matches[i + 1] = items[i];
    }
    free(items);
    return matches;
}

static char *completion_token_at_cursor(const char *line, int point) {
    if (!line || point < 0) {
        return strdup("");
    }

    size_t cursor = (size_t)point;
    size_t len = strlen(line);
    if (cursor > len) {
        cursor = len;
    }

    size_t start = cursor;
    while (start > 0 && !isspace((unsigned char)line[start - 1])) {
        start--;
    }

    char *tok = malloc(cursor - start + 1);
    if (!tok) {
        return NULL;
    }
    memcpy(tok, line + start, cursor - start);
    tok[cursor - start] = '\0';
    return tok;
}

static int token_is_first_word(const char *line, size_t cursor, const char *tok) {
    size_t tok_len = tok ? strlen(tok) : 0;
    size_t start = cursor >= tok_len ? cursor - tok_len : 0;
    for (size_t i = 0; i < start; i++) {
        if (!isspace((unsigned char)line[i])) {
            return 0;
        }
    }
    return 1;
}

static char **completion_matches_for_line(const char *line, size_t cursor, const struct shell_aliases *aliases) {
    char *tok = completion_token_at_cursor(line, (int)cursor);
    if (!tok) {
        return NULL;
    }

    if (cursor == 0) {
        free(tok);
        return NULL;
    }

    if (isspace((unsigned char)line[cursor - 1])) {
        int has_command = 0;
        for (size_t i = 0; i < cursor; i++) {
            if (!isspace((unsigned char)line[i])) {
                has_command = 1;
                break;
            }
        }
        free(tok);
        return has_command ? cwd_completion_matches(NULL, 1) : NULL;
    }

    if (tok[0] == '/') {
        char **matches = slash_command_completion_matches(tok);
        if (matches) {
            free(tok);
            return matches;
        }
    }

    if (tok[0] == '$') {
        char **matches = env_completion_matches(tok);
        free(tok);
        return matches;
    }

    if (strchr(tok, '/')) {
        int first_word = token_is_first_word(line, cursor, tok);
        char **matches = path_completion_matches(tok, !first_word);
        free(tok);
        return matches;
    }

    if (tok[0] == '-') {
        struct buffer view = {(char *)line, strlen(line)};
        char *cmd = command_before_cursor_token(&view, cursor);
        char **matches = NULL;
        if (cmd) {
            char *summary = load_or_build_command_options(cmd);
            matches = option_completion_matches_from_summary(summary, tok);
            free(summary);
        } else {
            fprintf(stderr, "\ncompletion: could not determine command before %s\n", tok);
        }
        free(cmd);
        free(tok);
        return matches;
    }

    char **matches = token_is_first_word(line, cursor, tok) ?
        command_completion_matches(tok, aliases) : cwd_completion_matches(tok, 1);
    free(tok);
    return matches;
}

static int terminal_width(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        return ws.ws_col;
    }
    return 80;
}

static void free_completion_matches(char **matches) {
    if (!matches) {
        return;
    }
    for (size_t i = 0; matches[i]; i++) {
        free(matches[i]);
    }
    free(matches);
}

static size_t completion_match_count(char **matches) {
    size_t count = 0;
    if (!matches) {
        return 0;
    }
    while (matches[count + 1]) {
        count++;
    }
    return count;
}

static int replace_completion_token(struct buffer *buf, size_t *cursor, const char *value) {
    const char *line = buf->data ? buf->data : "";
    char *tok = completion_token_at_cursor(line, (int)*cursor);
    if (!tok) {
        return 0;
    }
    size_t start = *cursor >= strlen(tok) ? *cursor - strlen(tok) : 0;
    size_t value_len = completion_insert_len(value);
    size_t tail_len = buf->len - *cursor;
    char *next = realloc(buf->data, start + value_len + tail_len + 1);
    if (!next) {
        free(tok);
        return 0;
    }
    buf->data = next;
    memmove(buf->data + start + value_len, buf->data + *cursor, tail_len + 1);
    memcpy(buf->data + start, value, value_len);
    buf->len = start + value_len + tail_len;
    *cursor = start + value_len;
    free(tok);
    return 1;
}

static size_t render_completion_menu(char **matches, size_t count, size_t selected) {
    int width = terminal_width();
    size_t max_len = 0;
    for (size_t i = 0; i < count; i++) {
        size_t len = strlen(completion_display_text(matches[i + 1]));
        if (len > max_len) {
            max_len = len;
        }
    }

    size_t col_width = max_len + 2;
    if (col_width < 8) {
        col_width = 8;
    }
    size_t cols = (size_t)width / col_width;
    if (cols == 0) {
        cols = 1;
    }
    if (cols > count) {
        cols = count;
    }
    size_t rows = (count + cols - 1) / cols;

    FILE *out = stderr;
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            size_t idx = c * rows + r;
            if (idx >= count) {
                continue;
            }
            const char *item = completion_display_text(matches[idx + 1]);
            if (idx == selected) {
                fputs(Q_COLOR_MENU_SELECTED, out);
            }
            fprintf(out, "%-*s", (int)col_width, item);
            if (idx == selected) {
                fputs(Q_COLOR_RESET, out);
            }
        }
        fputc('\n', out);
    }
    fflush(out);
    return rows;
}

static void clear_completion_menu(size_t rows) {
    FILE *out = stderr;
    if (rows > 0) {
        fprintf(out, "\033[%zuA", rows);
    }
    fputs("\r\033[J", out);
    fflush(out);
}

static int move_below_editor_input(const struct buffer *buf, size_t cursor, int line_no) {
    const char *data = buf && buf->data ? buf->data : "";
    size_t len = buf ? buf->len : 0;
    int width = terminal_width();
    if (width > 1) {
        width--;
    }
    size_t prompt_cols = (size_t)prompt_visible_cols(line_no);
    int end_row = 0;
    int end_col = 0;
    int cursor_row = 0;
    int cursor_col = 0;

    editor_position_for_index(data, len, len, prompt_cols, width, &end_row, &end_col);
    editor_position_for_index(data, len, cursor, prompt_cols, width, &cursor_row, &cursor_col);
    if (end_row > cursor_row) {
        fprintf(stderr, "\033[%dB", end_row - cursor_row);
    }
    fputs("\r\n", stderr);
    fflush(stderr);
    return end_row - cursor_row + 1;
}

static void move_back_to_editor_cursor_row(int rows_below_cursor) {
    if (rows_below_cursor > 0) {
        fprintf(stderr, "\033[%dA", rows_below_cursor);
        fflush(stderr);
    }
}

static void close_completion_menu(size_t rows, int rows_below_cursor) {
    clear_completion_menu(rows);
    move_back_to_editor_cursor_row(rows_below_cursor);
    show_terminal_cursor();
}

static int handle_visual_completion(int line_no, struct buffer *buf, size_t *cursor, const struct shell_aliases *aliases) {
    if (!buf || buf->len == 0) {
        return 0;
    }

    char **matches = completion_matches_for_line(buf->data ? buf->data : "", *cursor, aliases);
    size_t n = completion_match_count(matches);
    if (n == 0) {
        int rows_below_cursor = move_below_editor_input(buf, *cursor, line_no);
        fprintf(stderr, "no match!\n");
        fputs("\033[1A", stderr);
        fputs("\r\033[J", stderr);
        move_back_to_editor_cursor_row(rows_below_cursor);
        free_completion_matches(matches);
        return 0;
    }

    if (n == 1) {
        replace_completion_token(buf, cursor, matches[1]);
        free_completion_matches(matches);
        return 0;
    }

    hide_terminal_cursor();
    int rows_below_cursor = move_below_editor_input(buf, *cursor, line_no);
    size_t selected = 0;
    size_t rows = render_completion_menu(matches, n, selected);

    while (1) {
        unsigned char c = 0;
        if (read(STDIN_FILENO, &c, 1) != 1) {
            close_completion_menu(rows, rows_below_cursor);
            free_completion_matches(matches);
            return 0;
        }
        if (c == '\r' || c == '\n' || c == ' ') {
            close_completion_menu(rows, rows_below_cursor);
            replace_completion_token(buf, cursor, matches[selected + 1]);
            free_completion_matches(matches);
            return 0;
        }
        if (c == 7 || c == 27) {
            if (c == 27) {
                unsigned char c2 = 0;
                fd_set rfds;
                FD_ZERO(&rfds);
                FD_SET(STDIN_FILENO, &rfds);
                struct timeval tv = {0, 10000};
                int ready = select(STDIN_FILENO + 1, &rfds, NULL, NULL, &tv);
                if (ready <= 0 || !FD_ISSET(STDIN_FILENO, &rfds) || read(STDIN_FILENO, &c2, 1) != 1) {
                    close_completion_menu(rows, rows_below_cursor);
                    free_completion_matches(matches);
                    return 0;
                }
                if (c2 == '[') {
                    unsigned char c3 = 0;
                    if (read(STDIN_FILENO, &c3, 1) != 1) {
                        close_completion_menu(rows, rows_below_cursor);
                        free_completion_matches(matches);
                        return 0;
                    }
                    if (c3 == 'A') {
                        selected = selected == 0 ? n - 1 : selected - 1;
                    } else if (c3 == 'B') {
                        selected = (selected + 1) % n;
                    } else if (c3 == 'C') {
                        selected = (selected + 1) % n;
                    } else if (c3 == 'D') {
                        selected = selected == 0 ? n - 1 : selected - 1;
                    } else {
                        close_completion_menu(rows, rows_below_cursor);
                        free_completion_matches(matches);
                        return 0;
                    }
                    clear_completion_menu(rows);
                    rows = render_completion_menu(matches, n, selected);
                    continue;
                }
            }
            close_completion_menu(rows, rows_below_cursor);
            free_completion_matches(matches);
            return 0;
        }
        if (c == '\t') {
            selected = (selected + 1) % n;
        } else if (c == 127 || c == 8) {
            selected = selected == 0 ? n - 1 : selected - 1;
        } else {
            close_completion_menu(rows, rows_below_cursor);
            free_completion_matches(matches);
            if (isprint(c)) {
                size_t old_len = buf->len;
                char *next = realloc(buf->data, old_len + 2);
                if (next) {
                    buf->data = next;
                    if (old_len == 0) {
                        buf->data[0] = '\0';
                    }
                    memmove(buf->data + *cursor + 1, buf->data + *cursor, buf->len - *cursor + 1);
                    buf->data[*cursor] = (char)c;
                    buf->len++;
                    (*cursor)++;
                }
            }
            return 0;
        }
        clear_completion_menu(rows);
        rows = render_completion_menu(matches, n, selected);
    }
}

static int read_interactive_line(int line_no, struct history *hist, const struct shell_aliases *aliases, char **out, int color) {
    struct termios orig;
    struct buffer buf = {0};
    char *draft = NULL;
    ssize_t hist_pos = -1;
    size_t cursor = 0;
    int rendered_rows = 1;
    int rendered_cursor_row = 0;

    *out = NULL;
    if (!enable_raw_mode(&orig)) {
        return -1;
    }

    print_repl_prompt(line_no, color);
    fflush(stderr);

    while (1) {
        unsigned char c = 0;
        if (read(STDIN_FILENO, &c, 1) != 1) {
            if (errno == EINTR && request_interrupted) {
                request_interrupted = 0;
                request_interrupt_signal = 0;
                buf.len = 0;
                cursor = 0;
                if (buf.data) {
                    buf.data[0] = '\0';
                }
                fputc('\n', stderr);
                rendered_rows = 1;
                rendered_cursor_row = 0;
                print_repl_prompt(line_no, color);
                fflush(stderr);
                continue;
            }
            restore_termios(&orig);
            free(buf.data);
            free(draft);
            return -1;
        }

        if (c == '\r' || c == '\n') {
            fputc('\n', stderr);
            restore_termios(&orig);
            if (!append_char(&buf, '\0')) {
                free(buf.data);
                free(draft);
                return -1;
            }
            *out = buf.data;
            free(draft);
            return (int)buf.len - 1;
        }

        if (c == 4 && buf.len == 0) {
            fputc('\n', stderr);
            restore_termios(&orig);
            free(buf.data);
            free(draft);
            return -1;
        }
        if (c == 4) {
            if (cursor < buf.len) {
                delete_range(&buf, cursor, cursor + 1);
                refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
            }
            continue;
        }

        if (c == 1) {
            cursor = 0;
            refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
            continue;
        }
        if (c == 2) {
            if (cursor > 0) {
                cursor--;
                refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
            }
            continue;
        }
        if (c == 5) {
            cursor = buf.len;
            refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
            continue;
        }
        if (c == 6) {
            if (cursor < buf.len) {
                cursor++;
                refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
            }
            continue;
        }
        if (c == 11) {
            char *killed = buffer_slice(&buf, cursor, buf.len);
            clipboard_write_text(killed);
            free(killed);
            delete_range(&buf, cursor, buf.len);
            refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
            continue;
        }
        if (c == 12) {
            fputs("\033[H\033[2J", stderr);
            rendered_rows = 1;
            rendered_cursor_row = 0;
            refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
            continue;
        }
        if (c == 23) {
            size_t end = cursor;
            cursor = move_word_backward_pos(&buf, cursor);
            char *killed = buffer_slice(&buf, cursor, end);
            clipboard_write_text(killed);
            free(killed);
            delete_range(&buf, cursor, end);
            refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
            continue;
        }

        if (c == 22) {
            char *paste = clipboard_read_text();
            if (paste) {
                insert_text_at(&buf, &cursor, paste);
                free(paste);
                refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
            }
            continue;
        }

        if (c == '\t') {
            handle_visual_completion(line_no, &buf, &cursor, aliases);
            refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
            continue;
        }

        if (c == 16) {
            if (hist->len > 0) {
                if (hist_pos < 0) {
                    free(draft);
                    draft = strdup(buf.data ? buf.data : "");
                    hist_pos = (ssize_t)hist->len - 1;
                } else if (hist_pos > 0) {
                    hist_pos--;
                }
                if (hist_pos >= 0 && set_buffer_string(&buf, hist->items[hist_pos])) {
                    cursor = buf.len;
                }
            }
            refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
            continue;
        }

        if (c == 14) {
            if (hist_pos >= 0) {
                hist_pos++;
                if ((size_t)hist_pos >= hist->len) {
                    hist_pos = -1;
                    if (set_buffer_string(&buf, draft ? draft : "")) {
                        cursor = buf.len;
                    }
                } else if (set_buffer_string(&buf, hist->items[hist_pos])) {
                    cursor = buf.len;
                }
            }
            refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
            continue;
        }

        if (c == 127 || c == 8) {
            if (cursor > 0) {
                delete_range(&buf, cursor - 1, cursor);
                cursor--;
                refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
            }
            continue;
        }

        if (c == 27) {
            unsigned char seq[16];
            memset(seq, 0, sizeof(seq));
            if (read(STDIN_FILENO, &seq[0], 1) == 1 && seq[0] == '[' && read(STDIN_FILENO, &seq[1], 1) == 1) {
                if (seq[1] == 'A') {
                    if (hist->len > 0) {
                        if (hist_pos < 0) {
                            free(draft);
                            draft = strdup(buf.data ? buf.data : "");
                            hist_pos = (ssize_t)hist->len - 1;
                        } else if (hist_pos > 0) {
                            hist_pos--;
                        }
                        if (hist_pos >= 0 && set_buffer_string(&buf, hist->items[hist_pos])) {
                            cursor = buf.len;
                        }
                    }
                    refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
                } else if (seq[1] == 'B') {
                    if (hist_pos >= 0) {
                        hist_pos++;
                        if ((size_t)hist_pos >= hist->len) {
                            hist_pos = -1;
                            if (set_buffer_string(&buf, draft ? draft : "")) {
                                cursor = buf.len;
                            }
                        } else if (set_buffer_string(&buf, hist->items[hist_pos])) {
                            cursor = buf.len;
                        }
                    }
                    refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
                } else if (seq[1] == 'C') {
                    if (cursor < buf.len) {
                        cursor++;
                        refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
                    }
                } else if (seq[1] == 'D') {
                    if (cursor > 0) {
                        cursor--;
                        refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
                    }
                } else if (seq[1] == 'H') {
                    cursor = 0;
                    refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
                } else if (seq[1] == 'F') {
                    cursor = buf.len;
                    refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
                } else if (seq[1] == '3') {
                    if (read(STDIN_FILENO, &seq[2], 1) == 1) {
                        if (seq[2] == '~' && cursor < buf.len) {
                            delete_range(&buf, cursor, cursor + 1);
                            refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
                        } else if (seq[2] == ';' &&
                                   read(STDIN_FILENO, &seq[3], 1) == 1 &&
                                   read(STDIN_FILENO, &seq[4], 1) == 1 &&
                                   seq[3] == '6' && seq[4] == '~') {
                            size_t end = move_word_forward_pos(&buf, cursor);
                            char *killed = buffer_slice(&buf, cursor, end);
                            clipboard_write_text(killed);
                            free(killed);
                            delete_range(&buf, cursor, end);
                            refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
                        }
                    }
                } else if ((seq[1] >= '0' && seq[1] <= '9')) {
                    size_t i = 2;
                    while (i + 1 < sizeof(seq) && read(STDIN_FILENO, &seq[i], 1) == 1) {
                        if ((seq[i] >= 'A' && seq[i] <= 'Z') || seq[i] == '~' || seq[i] == 'u') {
                            break;
                        }
                        i++;
                    }
                    seq[sizeof(seq) - 1] = '\0';
                    if ((strstr((char *)seq, "1~") || strstr((char *)seq, "7~")) && cursor > 0) {
                        cursor = 0;
                        refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
                    } else if ((strstr((char *)seq, "4~") || strstr((char *)seq, "8~")) && cursor < buf.len) {
                        cursor = buf.len;
                        refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
                    } else if ((strstr((char *)seq, "1;5C") || strstr((char *)seq, "1;6C") ||
                                strstr((char *)seq, "1;10C") || strstr((char *)seq, "70;6u") ||
                                strstr((char *)seq, "102;6u") || strstr((char *)seq, "70;5u") ||
                                strstr((char *)seq, "102;5u")) && cursor < buf.len) {
                        cursor = move_word_forward_pos(&buf, cursor);
                        refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
                    } else if ((strstr((char *)seq, "1;5D") || strstr((char *)seq, "1;6D") ||
                                strstr((char *)seq, "1;10D") || strstr((char *)seq, "66;6u") ||
                                strstr((char *)seq, "98;6u") || strstr((char *)seq, "66;5u") ||
                                strstr((char *)seq, "98;5u")) && cursor > 0) {
                        cursor = move_word_backward_pos(&buf, cursor);
                        refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
                    } else if (strstr((char *)seq, "87;6u")) {
                        size_t end = move_word_forward_pos(&buf, cursor);
                        char *killed = buffer_slice(&buf, cursor, end);
                        clipboard_write_text(killed);
                        free(killed);
                        delete_range(&buf, cursor, end);
                        refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
                    }
                }
            } else if (seq[0] == 'O' && read(STDIN_FILENO, &seq[1], 1) == 1) {
                if (seq[1] == 'H') {
                    cursor = 0;
                    refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
                } else if (seq[1] == 'F') {
                    cursor = buf.len;
                    refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
                }
            }
            continue;
        }

        if (isprint(c)) {
            if (insert_char_at(&buf, cursor, (char)c)) {
                cursor++;
                refresh_editor_line(line_no, &buf, cursor, color, &rendered_rows, &rendered_cursor_row);
            }
        }
    }
}

static int repl_loop(int think_loud, int api_logging, int color, struct chat_history *chat) {
    char *line = NULL;
    size_t cap = 0;
    struct history hist = {0};
    struct shell_aliases aliases = {0};
    struct command_result last_command = {0};
    struct code_blocks blocks = {0};
    int last_status = 0;
    int line_no = 1;
    int have_command_status = 0;
    int interactive = isatty(STDIN_FILENO);
    int exit_requested = 0;

    if (interactive && !history_load_from_chat(&hist, chat)) {
        fprintf(stderr, "history: failed to load resumed session inputs\n");
    }
    load_shell_aliases(&aliases);

    while (1) {
        ssize_t n = 0;
        if (shutdown_signal) {
            break;
        }

        if (interactive) {
            if (have_command_status) {
                print_direct_tty_notice(&last_command);
                if (color) {
                    fprintf(stderr, "\n[Exit code %s%d" Q_COLOR_RESET "]\n",
                            last_status == 0 ? Q_COLOR_EXIT_OK : Q_COLOR_EXIT_FAIL, last_status);
                } else {
                    fprintf(stderr, "\n[Exit code %d]\n", last_status);
                }
                have_command_status = 0;
            }
            free(line);
            line = NULL;
            n = read_interactive_line(line_no, &hist, &aliases, &line, color);
        } else {
            n = getline(&line, &cap, stdin);
        }
        if (n < 0) {
            break;
        }
        line_no++;

        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
            line[--n] = '\0';
        }

        const char *trimmed = skip_ws(line);
        if (!*trimmed) {
            continue;
        }
        if (strcmp(trimmed, "/help") == 0) {
            print_repl_help(stdout, 1, think_loud, api_logging);
            continue;
        }
        if (strcmp(trimmed, "/keys") == 0) {
            print_key_bindings(stdout);
            continue;
        }
        if (strcmp(trimmed, "/show-system-prompt") == 0) {
            show_system_prompt(stdout);
            continue;
        }
        if (strncmp(trimmed, "/set-system-prompt", 18) == 0 &&
            (trimmed[18] == '\0' || isspace((unsigned char)trimmed[18]))) {
            const char *path = skip_ws(trimmed + 18);
            if (!*path) {
                print_system_prompt_file_help(stderr, "/set-system-prompt");
                continue;
            }
            if (!set_system_prompt_file(path, 0)) {
                continue;
            }
            fprintf(stderr, "/set-system-prompt: ");
            print_system_prompt_source(stderr);
            fprintf(stderr, "\n");
            continue;
        }
        if (strcmp(trimmed, "/exit") == 0) {
            exit_requested = 1;
            break;
        }
        if (strncmp(trimmed, "/llm-timeout", 12) == 0 &&
            (trimmed[12] == '\0' || isspace((unsigned char)trimmed[12]))) {
            const char *arg = skip_ws(trimmed + 12);
            long value = 0;
            if (!parse_llm_timeout_value(arg, &value)) {
                print_llm_timeout_help(stderr, "/llm-timeout");
                continue;
            }
            long previous = llm_timeout_seconds;
            llm_timeout_seconds = value;
            fprintf(stderr, "/llm-timeout: changing from ");
            print_llm_timeout_value(stderr, previous);
            fprintf(stderr, " to ");
            print_llm_timeout_value(stderr, llm_timeout_seconds);
            fprintf(stderr, "\n");
            continue;
        }
        if (strncmp(trimmed, "/llm-turn-limit", 15) == 0 &&
            (trimmed[15] == '\0' || isspace((unsigned char)trimmed[15]))) {
            const char *arg = skip_ws(trimmed + 15);
            long value = 0;
            if (!parse_llm_turn_limit_value(arg, &value)) {
                print_llm_turn_limit_help(stderr, "/llm-turn-limit");
                continue;
            }
            long previous = llm_turn_limit;
            llm_turn_limit = value;
            fprintf(stderr, "/llm-turn-limit: changing from %ld to %ld\n", previous, llm_turn_limit);
            continue;
        }
        if (strncmp(trimmed, "/think-loud", 11) == 0 &&
            (trimmed[11] == '\0' || isspace((unsigned char)trimmed[11]))) {
            const char *arg = skip_ws(trimmed + 11);
            int previous = think_loud;
            if (!*arg) {
                think_loud = !think_loud;
            } else if (strcmp(arg, "on") == 0) {
                think_loud = 1;
            } else if (strcmp(arg, "off") == 0) {
                think_loud = 0;
            } else {
                print_think_loud_help(stderr, "/think-loud");
                continue;
            }
            fprintf(stderr, "/think-loud: changing from %s to %s\n",
                    previous ? "on" : "off", think_loud ? "on" : "off");
            continue;
        }
        if (strncmp(trimmed, "/api-logging", 12) == 0 &&
            (trimmed[12] == '\0' || isspace((unsigned char)trimmed[12]))) {
            const char *arg = skip_ws(trimmed + 12);
            char *previous = strdup(api_logging_mode_name(api_logging));
            if (!previous) {
                fprintf(stderr, "/api-logging: memory allocation failed\n");
                continue;
            }
            if (!*arg) {
                free(api_logging_path);
                api_logging_path = NULL;
                api_logging = api_logging == API_LOGGING_NONE ? API_LOGGING_BOTH : API_LOGGING_NONE;
            } else if (!parse_api_logging_setting(arg, &api_logging)) {
                print_api_logging_help(stderr, "/api-logging");
                free(previous);
                continue;
            }
            fprintf(stderr, "/api-logging: changing from %s to %s\n",
                    previous, api_logging_mode_name(api_logging));
            free(previous);
            continue;
        }
        if (strncmp(trimmed, "/add-mcp-server", 15) == 0 &&
            (trimmed[15] == '\0' || isspace((unsigned char)trimmed[15]))) {
            const char *url = skip_ws(trimmed + 15);
            if (!*url) {
                print_add_mcp_server_help(stderr, "/add-mcp-server");
                continue;
            }
            if (mcp_add_server_command(url) == 0) {
                mcp_registry_free(&mcp_registry);
                if (!mcp_load_servers(&mcp_registry)) {
                    fprintf(stderr, "/add-mcp-server: failed to reload MCP config\n");
                } else {
                    mcp_probe_all(&mcp_registry);
                }
            }
            continue;
        }
        if (strncmp(trimmed, "/remove-mcp-server", 18) == 0 &&
            (trimmed[18] == '\0' || isspace((unsigned char)trimmed[18]))) {
            const char *name = skip_ws(trimmed + 18);
            if (!*name) {
                print_remove_mcp_server_help(stderr, "/remove-mcp-server");
                continue;
            }
            if (mcp_remove_server_command(name) == 0) {
                mcp_registry_free(&mcp_registry);
                if (!mcp_load_servers(&mcp_registry)) {
                    fprintf(stderr, "/remove-mcp-server: failed to reload MCP config\n");
                } else {
                    mcp_probe_all(&mcp_registry);
                }
            }
            continue;
        }
        if (strcmp(trimmed, "/list-mcp-servers") == 0) {
            mcp_list_servers_command();
            continue;
        }
        if (strncmp(trimmed, "/note", 5) == 0 &&
            (trimmed[5] == '\0' || isspace((unsigned char)trimmed[5]))) {
            const char *note = skip_ws(trimmed + 5);
            if (!*note) {
                fprintf(stderr,
                        "/note: missing text\n"
                        "Usage: /note text\n"
                        "Value:\n"
                        "  text                line to append to ~/.config/q/NOTES\n");
                continue;
            }
            append_note_line(note);
            continue;
        }
        if (strcmp(trimmed, "/truncate-context") == 0) {
            if (!chat || !chat->include_context) {
                fprintf(stderr, "/truncate-context: context is not enabled\n");
                continue;
            }
            if (!chat_history_truncate(chat)) {
                fprintf(stderr, "/truncate-context: failed\n");
            } else {
                fprintf(stderr, "/truncate-context: context is empty\n");
            }
            continue;
        }
        if (strcmp(trimmed, "/clear-completion-cache") == 0) {
            last_status = clear_completion_cache();
            continue;
        }
        if (trimmed[0] == '/') {
            char *slash_word = first_shell_word(trimmed);
            if (slash_word && !is_executable_word(slash_word)) {
                if (interactive) {
                    record_input_history(&hist, chat, trimmed);
                }
                fprintf(stderr, "unknown slash command: %s\n", slash_word);
                print_slash_commands_brief(stderr);
                free(slash_word);
                continue;
            }
            free(slash_word);
        }
        if (interactive) {
            record_input_history(&hist, chat, trimmed);
        }

        if (strcmp(trimmed, "??") == 0) {
            if (!last_command.valid) {
                fprintf(stderr, "??: no previous command output captured\n");
                continue;
            }
            if (last_command.direct_tty) {
                fprintf(stderr, "??: previous command was direct TTY command '%s'; output was not captured\n",
                        last_command.direct_tty_name && *last_command.direct_tty_name ? last_command.direct_tty_name : last_command.command);
                continue;
            }
            if (last_command.exit_code == 0) {
                fprintf(stderr, "??: previous command exited with 0; nothing failed!\n");
                continue;
            }
            char *question = build_failure_question(&last_command);
            if (!question) {
                fprintf(stderr, "??: memory allocation failed\n");
                continue;
            }
            last_status = perform_query(question, think_loud, api_logging, color, &blocks, chat, NULL, 0);
            free(question);
            continue;
        }

        if (trimmed[0] == '?' && isspace((unsigned char)trimmed[1])) {
            last_status = perform_query(skip_ws(trimmed + 1), think_loud, api_logging, color, &blocks, chat, NULL, 0);
            continue;
        }

        size_t block_index = 0;
        if (parse_block_reference(trimmed, &block_index)) {
            if (block_index >= blocks.len) {
                fprintf(stderr, "%s: no such executable block\n", trimmed);
                continue;
            }
            printf("%s\n\n", blocks.items[block_index]);
            fprintf(stderr, "[CMD]\n");
            last_status = run_shell_line(blocks.items[block_index], &last_command);
            if (interactive && !history_add_block_lines(&hist, blocks.items[block_index])) {
                fprintf(stderr, "history: memory allocation failed\n");
            }
            if (interactive && !chat_history_record_block_lines(chat, blocks.items[block_index])) {
                fprintf(stderr, "session history update failed\n");
            }
            have_command_status = 1;
            if (!interactive) {
                print_direct_tty_notice(&last_command);
                fprintf(stderr, "\n[Exit code %d]\n", last_status);
            }
            continue;
        }

        if (trimmed[0] == '!' && isspace((unsigned char)trimmed[1])) {
            const char *forced = skip_ws(trimmed + 1);
            char *expanded = expand_alias_line_recursive(forced, &aliases);
            fprintf(stderr, "[CMD]\n");
            last_status = run_shell_line(expanded ? expanded : forced, &last_command);
            free(expanded);
            have_command_status = 1;
            if (!interactive) {
                print_direct_tty_notice(&last_command);
                fprintf(stderr, "\n[Exit code %d]\n", last_status);
            }
            continue;
        }

        char *word = first_shell_word(trimmed);
        if (!word) {
            continue;
        }

        int done = 0;
        if (handle_blend_builtin(trimmed, word, &done)) {
            free(word);
            if (done) {
                break;
            }
            continue;
        }

        int is_alias = shell_aliases_contains(&aliases, word);
        if (should_execute_as_shell(word, &aliases)) {
            char *expanded = is_alias ? expand_alias_line_recursive(trimmed, &aliases) : NULL;
            fprintf(stderr, "[CMD]\n");
            last_status = run_shell_line(expanded ? expanded : trimmed, &last_command);
            free(expanded);
            have_command_status = 1;
            if (!interactive) {
                print_direct_tty_notice(&last_command);
                fprintf(stderr, "\n[Exit code %d]\n", last_status);
            }
        } else {
            last_status = perform_query(trimmed, think_loud, api_logging, color, &blocks, chat, NULL, 0);
        }

        free(word);
    }

    if (shutdown_signal) {
        chat_history_append_event(chat, "session interrupted; completed turns were saved", shutdown_signal);
        last_status = 128 + shutdown_signal;
    } else if (exit_requested) {
        chat_history_append_event(chat, "session closed with /exit", 0);
    }

    free(line);
    history_free(&hist);
    shell_aliases_free(&aliases);
    command_result_clear(&last_command);
    code_blocks_clear(&blocks);
    return last_status;
}

int main(int argc, char **argv) {
    atexit(restore_terminal_if_needed);

    int think_loud = 0;
    int api_logging = 0;
    int repl = 0;
    int color = 0;
    int keep_context = 0;
    int record_session = 0;
    int list_only = 0;
    int list_mcp_only = 0;
    const char *add_mcp_url = NULL;
    const char *remove_mcp_name = NULL;
    const char *resume_id = NULL;
    int first_arg = 1;

    while (first_arg < argc) {
        if (strcmp(argv[first_arg], "--help") == 0 || strcmp(argv[first_arg], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[first_arg], "--think-loud") == 0) {
            think_loud = 1;
        } else if (strcmp(argv[first_arg], "--keep-context") == 0) {
            keep_context = 1;
        } else if (strcmp(argv[first_arg], "--record-session") == 0) {
            record_session = 1;
        } else if (strcmp(argv[first_arg], "--list-sessions") == 0) {
            list_only = 1;
        } else if (strcmp(argv[first_arg], "--add-mcp-server") == 0) {
            if (first_arg + 1 >= argc) {
                print_add_mcp_server_help(stderr, "--add-mcp-server");
                return 2;
            }
            add_mcp_url = argv[++first_arg];
        } else if (strcmp(argv[first_arg], "--remove-mcp-server") == 0) {
            if (first_arg + 1 >= argc) {
                print_remove_mcp_server_help(stderr, "--remove-mcp-server");
                return 2;
            }
            remove_mcp_name = argv[++first_arg];
        } else if (strcmp(argv[first_arg], "--list-mcp-servers") == 0) {
            list_mcp_only = 1;
        } else if (strcmp(argv[first_arg], "--resume-session") == 0) {
            if (first_arg + 1 < argc && argv[first_arg + 1][0] != '-') {
                resume_id = argv[++first_arg];
            } else {
                resume_id = "last";
            }
            if (strcmp(resume_id, "last") != 0 &&
                !is_sequence_resume_id(resume_id) &&
                strncmp(resume_id, "session-", 8) != 0) {
                print_resume_session_help(stderr, "--resume-session");
                return 2;
            }
        } else if (strcmp(argv[first_arg], "--llm-timeout") == 0) {
            long value = 0;
            if (first_arg + 1 >= argc) {
                print_llm_timeout_help(stderr, "--llm-timeout");
                return 2;
            }
            if (!parse_llm_timeout_value(argv[++first_arg], &value)) {
                print_llm_timeout_help(stderr, "--llm-timeout");
                return 2;
            }
            llm_timeout_seconds = value;
        } else if (strcmp(argv[first_arg], "--llm-turn-limit") == 0) {
            long value = 0;
            if (first_arg + 1 >= argc) {
                print_llm_turn_limit_help(stderr, "--llm-turn-limit");
                return 2;
            }
            if (!parse_llm_turn_limit_value(argv[++first_arg], &value)) {
                print_llm_turn_limit_help(stderr, "--llm-turn-limit");
                return 2;
            }
            llm_turn_limit = value;
        } else if (strcmp(argv[first_arg], "--color") == 0) {
            color = 1;
        } else if (strcmp(argv[first_arg], "--api-logging") == 0) {
            if (first_arg + 1 >= argc) {
                print_api_logging_help(stderr, "--api-logging");
                return 2;
            }
            if (!parse_api_logging_setting(argv[++first_arg], &api_logging)) {
                print_api_logging_help(stderr, "--api-logging");
                return 2;
            }
        } else if (strcmp(argv[first_arg], "--system-prompt-file") == 0) {
            if (first_arg + 1 >= argc) {
                print_system_prompt_file_help(stderr, "--system-prompt-file");
                return 2;
            }
            if (!set_system_prompt_file(argv[++first_arg], 0)) {
                return 2;
            }
        } else if (strcmp(argv[first_arg], "--repl") == 0) {
            repl = 1;
        } else if (strcmp(argv[first_arg], "--blend") == 0) {
            fprintf(stderr, "--blend was renamed to --repl\n");
            return 2;
        } else {
            if (argv[first_arg][0] == '-') {
                fprintf(stderr, "unknown option: %s\n", argv[first_arg]);
                print_usage(argv[0]);
                return 2;
            }
            break;
        }
        first_arg++;
    }

    if (list_only) {
        return list_sessions();
    }

    if (!mcp_load_servers(&mcp_registry)) {
        fprintf(stderr, "failed to load MCP server config\n");
        return 1;
    }

    if (add_mcp_url) {
        int rc = mcp_add_server_command(add_mcp_url);
        mcp_registry_free(&mcp_registry);
        return rc;
    }

    if (remove_mcp_name) {
        int rc = mcp_remove_server_command(remove_mcp_name);
        mcp_registry_free(&mcp_registry);
        return rc;
    }

    if (list_mcp_only) {
        int rc = mcp_list_servers_command();
        mcp_registry_free(&mcp_registry);
        return rc;
    }

    mcp_probe_all(&mcp_registry);

    load_default_system_prompt_if_available();

    char *session_path = NULL;
    if (record_session && resume_id) {
        fprintf(stderr, "--record-session and --resume-session cannot be used together\n");
        return 2;
    }
    if (record_session) {
        session_path = new_session_path();
        if (!session_path) {
            fprintf(stderr, "failed to create session path\n");
            return 1;
        }
        printf("Session: %s\n", session_path);
    } else if (resume_id) {
        session_path = resume_session_path(resume_id);
        if (!session_path) {
            fprintf(stderr, "failed to resolve session path\n");
            return 1;
        }
    }

    struct chat_history chat = {
        .enabled = keep_context || session_path != NULL,
        .include_context = keep_context,
        .persist = session_path != NULL,
        .path = session_path
    };
    if (resume_id && !chat_history_load(&chat)) {
        fprintf(stderr, "failed to load context history from %s\n", chat.path);
        chat_history_free(&chat);
        free(session_path);
        return 1;
    }

    if (repl) {
        if (check_server_available() != 0) {
            chat_history_free(&chat);
            free(session_path);
            return 1;
        }
        install_shutdown_handlers();
        int rc = repl_loop(think_loud, api_logging, color, &chat);
        chat_history_free(&chat);
        free(session_path);
        return rc;
    }

    if (argc <= first_arg) {
        print_usage(argv[0]);
        chat_history_free(&chat);
        free(session_path);
        return 2;
    }

    if (check_server_available() != 0) {
        chat_history_free(&chat);
        free(session_path);
        return 1;
    }

    char *input1 = join_args(argc, argv, first_arg);
    if (!input1) {
        fprintf(stderr, "memory allocation failed\n");
        chat_history_free(&chat);
        free(session_path);
        return 1;
    }

    int rc = perform_query(input1, think_loud, api_logging, color, NULL, &chat, NULL, 0);
    chat_history_free(&chat);
    free(session_path);
    free(input1);
    return rc;
}
