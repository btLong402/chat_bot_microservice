import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Cấu hình logging
logging.basicConfig(
    level=logging.DEBUG,  # có thể đổi thành INFO khi chạy thực tế
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class MemoryManager:
    """Quản lý và lưu trữ lịch sử hội thoại, có thể tách riêng theo user."""

    _DEFAULT_USER_BUCKET = "__default__"

    def __init__(self, file_path: str = "chat_history.json", max_turns: int = 20):
        self.file_path = file_path
        self.max_turns = max_turns
        self._legacy_format = False
        self._store: Dict[str, List[Dict[str, str]]] = self._load_store()
        logging.info(f"MemoryManager initialized. File: {file_path}, Max turns: {max_turns}")

    def _load_store(self) -> Dict[str, List[Dict[str, str]]]:
        """Đọc dữ liệu từ file JSON."""
        if not os.path.exists(self.file_path):
            logging.debug("No existing history file found.")
            return {}

        try:
            with open(self.file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            logging.debug(f"Loaded data from {self.file_path}: type={type(data).__name__}")
        except Exception as e:
            logging.error(f"Failed to read {self.file_path}: {e}")
            return {}

        # Định dạng dictionary (đã có user bucket)
        if isinstance(data, dict):
            store: Dict[str, List[Dict[str, str]]] = {}
            for bucket, messages in data.items():
                if isinstance(messages, list):
                    store[bucket] = messages[-self.max_turns :]
            self._legacy_format = False
            logging.info(f"Loaded {len(store)} user buckets from JSON.")
            return store

        # Định dạng list (legacy format)
        if isinstance(data, list):
            self._legacy_format = True
            logging.warning("Legacy format detected (flat list). Converting to default bucket.")
            return {self._DEFAULT_USER_BUCKET: data[-self.max_turns :]}

        logging.warning("Unknown data format in JSON file.")
        self._legacy_format = False
        return {}

    def _bucket(self, user_id: Optional[str]) -> str:
        return user_id or self._DEFAULT_USER_BUCKET

    def save_history(self) -> None:
        """Lưu dữ liệu hiện tại vào file."""
        if not self._store:
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
                logging.info("Deleted history file because store is empty.")
            return

        directory = os.path.dirname(self.file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Chọn payload phù hợp
        if self._legacy_format and self._DEFAULT_USER_BUCKET in self._store and len(self._store) == 1:
            payload = self._store[self._DEFAULT_USER_BUCKET][-self.max_turns :]
            logging.debug("Saving in legacy format.")
        else:
            self._legacy_format = False
            payload = {
                bucket: messages[-self.max_turns :]
                for bucket, messages in self._store.items()
            }
            logging.debug(f"Saving {len(payload)} buckets.")

        try:
            with open(self.file_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            logging.info(f"History saved successfully to {self.file_path}")
        except Exception as e:
            logging.error(f"Failed to save history: {e}")

    def add_message(self, role: str, content: str, user_id: Optional[str] = None) -> None:
        """Thêm tin nhắn mới vào lịch sử."""
        bucket = self._bucket(user_id)
        message = {
            "role": role,
            "content": content,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        messages = self._store.setdefault(bucket, [])
        messages.append(message)
        self._store[bucket] = messages[-self.max_turns :]
        if bucket != self._DEFAULT_USER_BUCKET:
            self._legacy_format = False

        logging.debug(f"Added message: role={role}, user_id={user_id or 'default'}, total={len(messages)}")
        self.save_history()

    def get_conversation(self, user_id: Optional[str] = None) -> List[Tuple[str, str]]:
        """Lấy lại hội thoại theo user_id."""
        bucket = self._bucket(user_id)
        messages = self._store.get(bucket, [])
        if not messages and bucket != self._DEFAULT_USER_BUCKET:
            logging.debug(f"No history found for {user_id}, fallback to default bucket.")
            messages = self._store.get(self._DEFAULT_USER_BUCKET, [])

        conv = [(msg.get("role", ""), msg.get("content", "")) for msg in messages if isinstance(msg, dict)]
        logging.debug(f"Retrieved {len(conv)} messages for {user_id or 'default'}")
        return conv

    def clear_history(self, user_id: Optional[str] = None) -> None:
        """Xoá toàn bộ lịch sử của user (hoặc toàn cục)."""
        bucket = self._bucket(user_id)
        if bucket in self._store:
            del self._store[bucket]
            logging.info(f"Cleared history for {user_id or 'default'}")

        if not self._store:
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
                logging.info("All history cleared and file deleted.")
            self._legacy_format = False
        else:
            self.save_history()
