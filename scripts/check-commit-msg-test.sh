#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checker="$repo_root/scripts/check-commit-msg.sh"
message_file="$(mktemp)"
trap 'rm -f "$message_file"' EXIT

assert_accepts() {
    local description="$1"
    local message="$2"

    printf '%s\n' "$message" > "$message_file"
    if ! bash "$checker" "$message_file"; then
        printf 'expected acceptance: %s\n' "$description" >&2
        exit 1
    fi
}

assert_rejects() {
    local description="$1"
    local expected="$2"
    local message="$3"
    local output

    printf '%s\n' "$message" > "$message_file"
    if output="$(bash "$checker" "$message_file" 2>&1)"; then
        printf 'expected rejection: %s\n' "$description" >&2
        exit 1
    fi
    if [[ "$output" != *"$expected"* ]]; then
        printf 'wrong rejection for %s: %s\n' "$description" "$output" >&2
        exit 1
    fi
}

assert_accepts "conventional commit" $'feat(cli): add article export\n\nExplain the user-visible behavior and why it is needed.'
assert_accepts "breaking change" $'refactor(llm)!: replace provider boundary\n\nKeep the OpenAI-compatible configuration contract intact.'
assert_accepts "merge commit" "Merge branch 'main' into feature"
assert_accepts "fixup commit" "fixup! feat(cli): add article export"

assert_rejects "non-conventional subject" "Conventional Commit" \
    $'Add article export\n\nExplain the behavior.'
assert_rejects "long subject" "72 characters" \
    $'feat: add a deliberately overlong commit subject that cannot fit within the configured limit\n\nExplain the behavior.'
assert_rejects "missing body" "descriptive body" "fix: handle empty project file"
assert_rejects "long body line" "body lines must be at most 72 characters" \
    $'fix: handle empty project file\n\nThis deliberately overlong body line exceeds the repository limit by several characters.'
assert_rejects "agent attribution" "automated-agent attribution" \
    $'chore: configure repository hooks\n\nInstall both hook stages.\n\nGenerated-by: Amp'
assert_rejects "Amp thread ID" "session or thread IDs" \
    $'chore: configure repository hooks\n\nInstall both hook stages.\n\nThread: T-01a0722f-e3be-765a-9f5d-5f96cb5acf7e'

printf 'commit-message policy tests passed\n'
