# LLM interleaved with Terminal

A proof of concept of interleaving the terminal/shell with the LLM. so you can do day to day shell commands and if stuck or want more clarification, in the same prompt the users post their question to the LLM, and get answers there, thus no need to hop to the browser. kind of nearby senior who can help clarify or provide examples for what you are trying to do. Examples (after launching it in repl mode with q --repl )

$ q --repl --color

```text
1 $ show the last 100 nginx service logs without paging
```

```text
2 $ restart nginx and verify it is listening on ports 80 and 443
```

```text
3 $ list failed systemd units and show commands to inspect each failure
```

```text
4 $ show disk usage by top-level directories under /var
```

```text
5 $ find files larger than 1G under /home modified in the last 7 days
```

```text
6 $ check apt package holds and pending security upgrades
```

```text
7 $ create a systemd timer that runs /usr/local/bin/backup.sh daily at 02:30
```

```text
8 $ write an rsync command to mirror /srv/data to backup@example:/backups/data preserving permissions and deleting removed files
```

```text
9 $ ? top processes by memory with ps, sorted descending
```

and many others like:

```text
show ufw commands to allow ssh, http, and https then enable the firewall
diagnose DNS resolution problems using resolvectl, dig, and systemd-resolved logs
show commands to list users with sudo access
generate a logrotate config for /var/log/myapp/*.log keeping 14 compressed daily logs
check memory pressure and swap usage from the command line
show commands to inspect SMART health for /dev/sda
```

Hope you got the drift. It is rough and rather large with ~7k of vibed code, and I had been using it as a daily driver.

Note that the output of queries will depend on the LLM used.

Sometimes q will be confused with a query for a command (i.e. who or which), usually this can be workaround by Capitalizing the first word of a query.

Still you can use ! to force a shell execution or ? to force a query.

If an LLM was asked for a shell/bash/zsh examples and these were fenced, then these are detected and are numbered from 1 to the last one. you can then type the number followed by dot (i.e. 2.) which will execute it without typing it.

for help use slash command /help which will show other commands.

In summary q is a small vibed (using codex) C CLI named `q` that sends all command-line arguments as one `input1` string to a local OpenAI-compatible Responses API using a Linux sysadmin system prompt. Responses are streamed to stdout as they arrive.

Note that again, this is a proof of concept rather than finished work, and waiting until it is ready will take forever, thus opting to publish the code so other smarter people can have fun and run with it.

Below is the start of LLM generated content

## Build

```bash
make
```

For a stripped release binary:

```bash
make release
```

Install to `/usr/local/bin` by default:

```bash
sudo make install
```

Requires libcurl development headers:

```bash
sudo apt install build-essential libcurl4-openssl-dev
```

Colors can be overridden at compile time with C macros such as `Q_COLOR_LLM_TEXT`, `Q_COLOR_CODE_EMULATOR`, `Q_COLOR_CODE_TTY`, `Q_COLOR_PROMPT_LINE_NO`, `Q_COLOR_EXIT_OK`, `Q_COLOR_EXIT_FAIL`, `Q_COLOR_MENU_SELECTED`, `Q_COLOR_TOOL_CONFIRM`, and `Q_COLOR_RESET`.

## Run

Usage:

```bash
./q [--repl] [--keep-context] [--record-session] [--resume-session [id|last]] [--list-sessions] [--add-mcp-server url] [--remove-mcp-server name] [--list-mcp-servers] [--llm-timeout seconds] [--llm-turn-limit count] [--api-logging none|query|response|both|path] [--system-prompt-file filepath] [--think-loud] [--color] [query words...]
```

```bash
export OPENAI_API_KEY='your_api_key'
./q --repl
```

For local OpenAI-compatible servers that do not require auth, leave `OPENAI_API_KEY` unset.

If `LLM_SERVER` is unset or empty, `q` uses `127.0.0.1`.
If `LLM_PORT` is unset or empty, `q` uses `8080`.
The default endpoint is `http://127.0.0.1:8080/v1/responses`.

Also One-shot mode is also available when you do not want to enter the REPL:

```bash
./q find which process is using port 8080
```

```bash
LLM_SERVER=127.0.0.1:8000 ./q check nginx logs last 50 lines
```

```bash
LLM_SERVER=http://127.0.0.1:8000/v1/responses ./q check nginx logs last 50 lines
```

```bash
LLM_PORT=1234 ./q check nginx logs last 50 lines
```

HTTP MCP servers:

```bash
./q --add-mcp-server http://127.0.0.1:3000/mcp
```

```bash
./q --list-mcp-servers
```

```bash
./q --remove-mcp-server name
```

Registered HTTP MCP servers are stored in `~/.config/q/mcp-servers.tsv`. `--add-mcp-server` reads the server name from the MCP `initialize` response. At startup, `q` probes registered servers with `initialize` and `tools/list`; live tools are appended to the builtin `get_time`, `read_file`, and `write_file` tools sent to the model.

Limit LLM/tool follow-up turns:

```bash
./q --llm-turn-limit 4 check something
```

API logging:

```bash
./q --api-logging query find which process is using port 8080
```

Modes are `none`, `query`, `response`, `both`, or an appendable file path. Default: `none`.

By default, streamed reasoning/thinking output is hidden and shown as animated dots on stderr. To show it:

```bash
./q --repl --think-loud
```

REPL slash commands include `/help`, `/keys`, `/show-system-prompt`, `/set-system-prompt path`, `/llm-timeout seconds`, `/llm-turn-limit count`, `/think-loud on|off`, `/api-logging none|query|response|both|path`, `/add-mcp-server url`, `/remove-mcp-server name`, `/list-mcp-servers`, `/truncate-context`, `/clear-completion-cache`, `/note text`, and `/exit`.

On startup, `q` checks that the configured local LLM server is reachable before entering query or REPL mode.

With `--keep-context`, each new LLM request includes previous user/assistant turns from the current `q` run. Without it, each request sends only the current input.

Session files:

```bash
./q --record-session --repl
```

```bash
./q --list-sessions
```

```bash
./q --resume-session 1 --repl
```

```bash
./q --resume-session --repl
```

Recorded sessions are stored in `~/.config/q/sessions/session-<timestamp>`. `--list-sessions` prints a numeric ID; use that number with `--resume-session N`, resume by timestamp/session id, or use `--resume-session last`. If no id is supplied to `--resume-session`, `last` is used. Recording/resuming a session does not automatically enable context; add `--keep-context` when you want prior turns sent to the LLM.

Interactive REPL prompts look like `1 $        `, `2 $        `, etc. There are 8 spaces after `$`. With `--color`, the line number is dark red, exit code `0` is green, and nonzero exit codes are red.

Interactive REPL line editing:

```text
C-p  previous history item
C-n  next history item
C-a  beginning of line
C-e  end of line
C-k  delete to end of line
C-w  delete previous word
C-d  delete character at cursor
Del  delete character at cursor
C-f  forward character
C-b  backward character
C-l  clear screen
Home beginning of line
End  end of line
Tab  show completion suggestions
```

Long input lines redraw across wrapped terminal rows without duplicating the line.

Tab behavior:

```text
After at least one typed character: complete slash commands, aliases, environment variables, directories, executable `./` files, or command arguments depending on cursor position.
On a token starting with - or --: infer the command before the option, read its man page, and ask the LLM to summarize relevant options below the input. If there is no manual entry, use the command's `--help` output instead.
```

Command option summaries are cached globally in `~/.config/q-completions/`. Later `q` instances reuse the cached summary and filter it by the current `-` or `--` prefix instead of asking the LLM again.

Typing more text, including a space, redraws the input and clears the suggestion area.

REPL escapes:

```bash
? which command shows open ports
```

```bash
! which bash
```

`?` forces the line to the LLM. `!` forces the line to the shell. `/exit` exits the REPL. `/clear-completion-cache` removes cached command option completions.

After a shell command runs, `q` captures the command, combined stdout/stderr output, and exit code. Enter `??` to ask the LLM:

```text
the following command failed: 'the command', with output of 'the output', and exit code 'the exit code', what is the problem, or solution?
```

When an LLM reply contains fenced `bash`, `sh`, `shell`, or `zsh` code blocks, REPL mode lists them as executable blocks:

````text
1.
```bash
pwd
```
````

Type the number with a dot to execute it:

```text
1.
```

`q` prints the selected command text, then a blank line, before executing it.

Optional model override:

```bash
OPENAI_MODEL=gpt-5.5 ./q check nginx logs last 50 lines
```
