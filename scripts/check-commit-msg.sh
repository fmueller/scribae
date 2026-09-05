#!/usr/bin/env bash
set -euo pipefail

message_file="${1:?usage: check-commit-msg.sh MESSAGE_FILE}"
subject="$(sed -n '1p' "$message_file" | tr -d '\r')"

case "$subject" in
    Merge\ * | Revert\ * | fixup!\ * | squash!\ *) exit 0 ;;
esac

if ((${#subject} > 72)); then
    printf 'commit subject must be at most 72 characters\n' >&2
    exit 1
fi

if [[ ! "$subject" =~ ^(feat|fix|refactor|docs|test|chore|build|perf|ci)(\([a-z0-9._-]+\))?\!?:\ .+ ]]; then
    printf 'commit subject must use Conventional Commit format\n' >&2
    exit 1
fi

if [[ -n "$(sed -n '2p' "$message_file" | tr -d '\r')" ]] || \
    ! tail -n +3 "$message_file" | grep -q '[^[:space:]]'; then
    printf 'commit message must include a blank-line-separated descriptive body\n' >&2
    exit 1
fi

if tail -n +3 "$message_file" | awk 'length($0) > 72 { exit 1 }'; then
    :
else
    printf 'commit body lines must be at most 72 characters\n' >&2
    exit 1
fi

if grep -Eiq \
    '(co-authored-by|assisted-by|generated-by):.*(amp|agent|claude|codex|copilot|chatgpt|cursor)|generated (with|by).*(amp|agent|claude|codex|copilot|chatgpt|cursor)' \
    "$message_file"; then
    printf 'commit message must not contain automated-agent attribution\n' >&2
    exit 1
fi

if grep -Eiq \
    'T-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|(session|thread)[-_ ]?id:' \
    "$message_file"; then
    printf 'commit message must not contain session or thread IDs\n' >&2
    exit 1
fi
