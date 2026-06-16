#!/bin/bash
INPUT=$(cat)
CMD=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('command',''))" 2>/dev/null)

[[ "$CMD" == *"git commit"* ]] || exit 0

if echo "$CMD" | grep -q "Co-Authored-By"; then
    echo "ERROR: Remove Co-Authored-By from commit message (project convention: short one-liners only)" >&2
    exit 1
fi

exit 0
