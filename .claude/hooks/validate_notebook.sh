#!/bin/bash
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)

[[ "$FILE_PATH" == *.ipynb ]] || exit 0

if python3 -m json.tool "$FILE_PATH" > /dev/null 2>&1; then
    echo "Notebook OK: $FILE_PATH"
else
    echo "WARNING: $FILE_PATH appears to be malformed JSON — notebook may be corrupted"
fi

exit 0
