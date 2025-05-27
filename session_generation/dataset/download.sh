#!/bin/bash

# === 檔案格式範例 ===
# 每一行為：<FILE_ID> <FILENAME>
# 例如：
# 1AbcDefGhIJKL file1.zip
# 2XyZ987654321 file2.pdf
LIST_FILE="file_list.txt"

# 建立暫存檔
COOKIE_FILE=$(mktemp)
HTML_FILE=$(mktemp)

# 下載函式
download_file() {
  FILE_ID=$1
  FILE_NAME=$2

  echo "Downloading ${FILE_NAME}..."

  curl -c $COOKIE_FILE -s -L \
    "https://drive.google.com/uc?export=download&id=${FILE_ID}" > $HTML_FILE

  CONFIRM=$(awk '/download/ {print $NF}' $HTML_FILE | sed 's/.*confirm=\(.*\)&id=.*/\1/')

  curl -Lb $COOKIE_FILE \
    "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" \
    -o "${FILE_NAME}"

  echo "✔ Finished: ${FILE_NAME}"
}

# 主程式：從清單逐行讀取
while read -r FILE_ID FILE_NAME; do
  [[ -z "$FILE_ID" || -z "$FILE_NAME" ]] && continue
  download_file "$FILE_ID" "$FILE_NAME"
done < "$LIST_FILE"

# 清理暫存
rm -f $COOKIE_FILE $HTML_FILE
