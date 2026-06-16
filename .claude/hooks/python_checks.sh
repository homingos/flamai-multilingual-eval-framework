#!/bin/bash
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)

[[ "$FILE_PATH" == *.py ]] || exit 0

if command -v black &>/dev/null; then
    black "$FILE_PATH" 2>&1
fi

if command -v flake8 &>/dev/null; then
    flake8 "$FILE_PATH" 2>&1
fi

exit 0
