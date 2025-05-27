#!/bin/bash

# 使用方式:
# ./gdrive_batch_download.sh file_list.txt --output-dir ./downloads

# 參數
LIST_FILE="$1"
shift
OUTPUT_DIR="."

# 處理選項
while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "❌ 未知參數: $1"
      exit 1
      ;;
  esac
done

# 建立儲存資料夾
mkdir -p "$OUTPUT_DIR"

# 建立暫存檔
COOKIE_FILE=$(mktemp)
HTML_FILE=$(mktemp)

download_file() {
  FILE_ID=$1
  FILE_NAME=$2
  DEST_PATH="${OUTPUT_DIR}/${FILE_NAME}"

  echo "📥 Downloading ${FILE_NAME}..."

  curl -c $COOKIE_FILE -s -L \
    "https://drive.google.com/uc?export=download&id=${FILE_ID}" > $HTML_FILE

  CONFIRM=$(awk '/download/ {print $NF}' $HTML_FILE | sed 's/.*confirm=\(.*\)&id=.*/\1/')

  curl -Lb $COOKIE_FILE \
    "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" \
    -o "$DEST_PATH"

  echo "✔ Saved to: $DEST_PATH"
}

# 主程式
while read -r FILE_ID FILE_NAME; do
  [[ -z "$FILE_ID" || -z "$FILE_NAME" ]] && continue
  download_file "$FILE_ID" "$FILE_NAME"
done < "$LIST_FILE"

# 清理暫存
rm -f $COOKIE_FILE $HTML_FILE
