from __future__ import annotations

import json
import base64
import os
import random
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    httpx = None
    HAS_HTTPX = False

RED = "\033[31m"
RESET = "\033[0m"

ITEM_TEXT = 1
MESSAGE_TYPE_USER = 1
MESSAGE_TYPE_BOT = 2
MESSAGE_STATE_FINISH = 2
WEIXIN_MAX_MESSAGE_LEN = 4000
WEIXIN_CHANNEL_VERSION = "2.1.1"
WEIXIN_APP_CLIENT_VERSION = (2 << 16) | (1 << 8) | 1
WEIXIN_BASE_INFO = {"channel_version": WEIXIN_CHANNEL_VERSION}

def _default_prompt() -> str:
    return "You > "

def _default_send(text: str) -> None:
    print(text)

def _default_info(text: str) -> None:
    print(text)

@dataclass
class InboundMessage:
    text: str
    sender_id: str
    channel: str = ""
    account_id: str = ""
    peer_id: str = ""
    is_group: bool = False
    media: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)

# 通道帐号设置
@dataclass
class ChannelAccount:
    channel: str
    account_id: str
    token: str = ""
    config: dict = field(default_factory=dict)

# ---------------------------------------------------------------------------
# Channel 类
# ---------------------------------------------------------------------------

# 抽象基类
class Channel(ABC):
    name: str = "unknown"

    @abstractmethod
    def receive(self) -> InboundMessage | None: ...

    @abstractmethod
    def send(self, to: str, text: str, **kwargs: Any) -> bool: ...

    def close(self) -> None:
        pass

# CLI Channel
class CLIChannel(Channel):
    name = "cli"

    def __init__(self, prompt_func: Callable[[], str] | None = None, send_func: Callable[[str], None] | None = None) -> None:
        self.account_id = "cli-local"
        self._prompt_func = prompt_func or _default_prompt
        self._send_func = send_func or _default_send
        self._input_allowed = threading.Event()
        self._input_allowed.set()

    def allow_input(self) -> None:
        self._input_allowed.set()

    def receive(self) -> InboundMessage | None:
        self._input_allowed.wait()
        self._input_allowed.clear()
        try:
            text = input(self._prompt_func()).strip()
        except (KeyboardInterrupt, EOFError):
            return None
        if not text:
            self._input_allowed.set()
            return None
        return InboundMessage(
            text=text, sender_id="cli-user", channel="cli",
            account_id=self.account_id, peer_id="cli-user",
        )

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        self._send_func(text)
        return True

# FeishuChannel - 基于 webhook
class DefaultChannel(Channel):
    name = "default"

    def __init__(self, send_func: Callable[[str], None] | None = None) -> None:
        self._send_func = send_func or _default_send

    def receive(self) -> InboundMessage | None:
        return None

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        self._send_func(text)
        return True

class FeishuChannel(Channel):
    name = "feishu"

    def __init__(self, account: ChannelAccount) -> None:
        if not HAS_HTTPX:
            raise RuntimeError("FeishuChannel requires httpx: pip install httpx")
        self.account_id = account.account_id
        self.app_id = account.config.get("app_id", "")
        self.app_secret = account.config.get("app_secret", "")
        self._encrypt_key = account.config.get("encrypt_key", "")
        self._bot_open_id = account.config.get("bot_open_id", "")
        is_lark = account.config.get("is_lark", False)
        self.api_base = ("https://open.larksuite.com/open-apis" if is_lark
                         else "https://open.feishu.cn/open-apis")
        self._tenant_token: str = ""
        self._token_expires_at: float = 0.0
        self._http = httpx.Client(timeout=15.0)

    def _refresh_token(self) -> str:
        if self._tenant_token and time.time() < self._token_expires_at:
            return self._tenant_token
        try:
            resp = self._http.post(
                f"{self.api_base}/auth/v3/tenant_access_token/internal",
                json={"app_id": self.app_id, "app_secret": self.app_secret},
            )
            data = resp.json()
            if data.get("code") != 0:
                print(f"{RED}[feishu] Token error: {data.get('msg', '?')}{RESET}")
                return ""
            self._tenant_token = data.get("tenant_access_token", "")
            self._token_expires_at = time.time() + data.get("expire", 7200) - 300
            return self._tenant_token
        except Exception as exc:
            print(f"{RED}[feishu] Token error: {exc}{RESET}")
            return ""

    def _bot_mentioned(self, event: dict) -> bool:
        for m in event.get("message", {}).get("mentions", []):
            mid = m.get("id", {})
            if isinstance(mid, dict) and mid.get("open_id") == self._bot_open_id:
                return True
            if isinstance(mid, str) and mid == self._bot_open_id:
                return True
            if m.get("key") == self._bot_open_id:
                return True
        return False

    def _parse_content(self, message: dict) -> tuple[str, list]:
        msg_type = message.get("msg_type", "text")
        raw = message.get("content", "{}")
        try:
            content = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            return "", []

        media: list[dict] = []
        if msg_type == "text":
            return content.get("text", ""), media
        if msg_type == "post":
            texts: list[str] = []
            for lc in content.values():
                if not isinstance(lc, dict):
                    continue
                title = lc.get("title", "")
                if title:
                    texts.append(title)
                for para in lc.get("content", []):
                    for node in para:
                        tag = node.get("tag")
                        if tag == "text":
                            texts.append(node.get("text", ""))
                        elif tag == "a":
                            texts.append(node.get("text", "") + " " + node.get("href", ""))
            return "\n".join(texts), media
        if msg_type == "image":
            key = content.get("image_key", "")
            if key:
                media.append({"type": "image", "key": key})
            return "[image]", media
        return "", media

    def parse_event(self, payload: dict, token: str = "") -> InboundMessage | None:
        """解析飞书事件回调。使用简单的 token 校验进行验证。"""
        if self._encrypt_key and token and token != self._encrypt_key:
            print(f"{RED}[feishu] Token verification failed{RESET}")
            return None
        if "challenge" in payload:
            _default_info(f"[feishu] Challenge: {payload['challenge']}")
            return None

        event = payload.get("event", {})
        message = event.get("message", {})
        sender = event.get("sender", {}).get("sender_id", {})
        user_id = sender.get("open_id", sender.get("user_id", ""))
        chat_id = message.get("chat_id", "")
        chat_type = message.get("chat_type", "")
        is_group = chat_type == "group"

        if is_group and self._bot_open_id and not self._bot_mentioned(event):
            return None

        text, media = self._parse_content(message)
        if not text:
            return None

        return InboundMessage(
            text=text, sender_id=user_id, channel="feishu",
            account_id=self.account_id,
            peer_id=user_id if chat_type == "p2p" else chat_id,
            media=media, is_group=is_group, raw=payload,
        )

    def receive(self) -> InboundMessage | None:
        return None

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        token = self._refresh_token()
        if not token:
            return False
        try:
            resp = self._http.post(
                f"{self.api_base}/im/v1/messages",
                params={"receive_id_type": "chat_id"},
                headers={"Authorization": f"Bearer {token}"},
                json={"receive_id": to, "msg_type": "text",
                      "content": json.dumps({"text": text})},
            )
            data = resp.json()
            if data.get("code") != 0:
                print(f"{RED}[feishu] Send: {data.get('msg', '?')}{RESET}")
                return False
            return True
        except Exception as exc:
            print(f"{RED}[feishu] Send: {exc}{RESET}")
            return False

    def close(self) -> None:
        self._http.close()

# ---------------------------------------------------------------------------
# Channel 管理
# ---------------------------------------------------------------------------

class WeixinPersonalChannel(Channel):
    name = "weixin"
    background_receive = True

    def __init__(self, account: ChannelAccount) -> None:
        if not HAS_HTTPX:
            raise RuntimeError("WeixinPersonalChannel requires httpx: pip install httpx")
        self.account_id = account.account_id
        self.base_url = account.config.get("base_url", "https://ilinkai.weixin.qq.com").rstrip("/")
        self.route_tag = str(account.config.get("route_tag", "") or "").strip()
        self.allow_from = list(account.config.get("allow_from", ["*"]) or ["*"])
        self.poll_timeout = int(account.config.get("poll_timeout", 35) or 35)
        state_dir = account.config.get("state_dir", "")
        self.state_dir = Path(state_dir).expanduser() if state_dir else Path(".state") / "weixin"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._token = str(account.config.get("token", "") or "")
        self._get_updates_buf = ""
        self._context_tokens: dict[str, str] = {}
        self._processed_ids: dict[str, float] = {}
        self._closed = False
        self._http = httpx.Client(
            timeout=httpx.Timeout(self.poll_timeout + 10, connect=30),
            follow_redirects=True,
        )
        self._load_state()

    @staticmethod
    def _random_wechat_uin() -> str:
        return base64.b64encode(str(int.from_bytes(os.urandom(4), "big")).encode()).decode()

    def _headers(self, auth: bool = True) -> dict[str, str]:
        headers = {
            "X-WECHAT-UIN": self._random_wechat_uin(),
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
            "iLink-App-Id": "bot",
            "iLink-App-ClientVersion": str(WEIXIN_APP_CLIENT_VERSION),
        }
        if auth and self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if self.route_tag:
            headers["SKRouteTag"] = self.route_tag
        return headers

    def _state_file(self) -> Path:
        return self.state_dir / "account.json"

    def _load_state(self) -> None:
        path = self._state_file()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._token = self._token or data.get("token", "")
            self._get_updates_buf = data.get("get_updates_buf", "")
            self._context_tokens = {
                str(k): str(v)
                for k, v in data.get("context_tokens", {}).items()
                if str(k).strip() and str(v).strip()
            }
            if data.get("base_url"):
                self.base_url = str(data["base_url"]).rstrip("/")
        except (OSError, json.JSONDecodeError, AttributeError):
            return

    def _save_state(self) -> None:
        data = {
            "token": self._token,
            "get_updates_buf": self._get_updates_buf,
            "context_tokens": self._context_tokens,
            "base_url": self.base_url,
        }
        self._state_file().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _api_get(self, endpoint: str, params: dict | None = None, auth: bool = True, base_url: str | None = None) -> dict:
        resp = self._http.get(
            f"{(base_url or self.base_url).rstrip('/')}/{endpoint}",
            params=params,
            headers=self._headers(auth=auth),
        )
        resp.raise_for_status()
        return resp.json()

    def _api_post(self, endpoint: str, body: dict | None = None, auth: bool = True) -> dict:
        payload = body or {}
        payload.setdefault("base_info", WEIXIN_BASE_INFO)
        resp = self._http.post(
            f"{self.base_url}/{endpoint}",
            json=payload,
            headers=self._headers(auth=auth),
        )
        resp.raise_for_status()
        return resp.json()

    def login(self, force: bool = False) -> bool:
        if force:
            self._token = ""
            self._get_updates_buf = ""
            self._context_tokens = {}
            try:
                self._state_file().unlink()
            except FileNotFoundError:
                pass
        if self._token:
            return True

        qrcode_id, scan_url = self._fetch_qr_code()
        self._print_qr_code(scan_url)
        poll_base_url = self.base_url
        refresh_count = 0
        while not self._closed:
            data = self._api_get(
                "ilink/bot/get_qrcode_status",
                params={"qrcode": qrcode_id},
                auth=False,
                base_url=poll_base_url,
            )
            status = data.get("status", "")
            if status == "confirmed":
                token = data.get("bot_token", "")
                if not token:
                    print(f"{RED}[weixin] Login confirmed but bot_token is missing{RESET}")
                    return False
                self._token = token
                if data.get("baseurl"):
                    self.base_url = str(data["baseurl"]).rstrip("/")
                self._save_state()
                print(f"[weixin] Login successful: user={data.get('ilink_user_id', '')}")
                return True
            if status == "scaned_but_redirect":
                redirect_host = str(data.get("redirect_host", "") or "").strip()
                if redirect_host:
                    poll_base_url = redirect_host if redirect_host.startswith(("http://", "https://")) else f"https://{redirect_host}"
            elif status == "expired":
                refresh_count += 1
                if refresh_count > 3:
                    print(f"{RED}[weixin] QR code expired too many times{RESET}")
                    return False
                qrcode_id, scan_url = self._fetch_qr_code()
                self._print_qr_code(scan_url)
                poll_base_url = self.base_url
            time.sleep(1)
        return False

    def _fetch_qr_code(self) -> tuple[str, str]:
        data = self._api_get("ilink/bot/get_bot_qrcode", params={"bot_type": "3"}, auth=False)
        qrcode_id = data.get("qrcode", "")
        if not qrcode_id:
            raise RuntimeError(f"Failed to get Weixin QR code: {data}")
        return qrcode_id, data.get("qrcode_img_content", "") or qrcode_id

    @staticmethod
    def _print_qr_code(url: str) -> None:
        try:
            import qrcode

            qr = qrcode.QRCode(border=1)
            qr.add_data(url)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
        except ImportError:
            print(f"\n[weixin] Login URL: {url}\n")

    def _allowed(self, user_id: str) -> bool:
        return "*" in self.allow_from or user_id in self.allow_from

    def _parse_message(self, msg: dict) -> InboundMessage | None:
        if msg.get("message_type") == MESSAGE_TYPE_BOT:
            return None
        msg_id = str(msg.get("message_id", "") or msg.get("seq", ""))
        if not msg_id:
            msg_id = f"{msg.get('from_user_id', '')}_{msg.get('create_time_ms', '')}"
        if msg_id in self._processed_ids:
            return None
        self._processed_ids[msg_id] = time.time()
        if len(self._processed_ids) > 1000:
            oldest = sorted(self._processed_ids, key=self._processed_ids.get)[:200]
            for key in oldest:
                self._processed_ids.pop(key, None)

        user_id = str(msg.get("from_user_id", "") or "")
        if not user_id or not self._allowed(user_id):
            return None
        context_token = str(msg.get("context_token", "") or "")
        if context_token:
            self._context_tokens[user_id] = context_token
            self._save_state()

        parts: list[str] = []
        media: list[dict[str, str]] = []
        for item in msg.get("item_list", []) or []:
            if item.get("type") == ITEM_TEXT:
                text = (item.get("text_item") or {}).get("text", "")
                if text:
                    parts.append(text)
            elif item.get("type") == 2:
                parts.append("[image]")
                media.append({"type": "image"})
            elif item.get("type") == 3:
                voice_text = ((item.get("voice_item") or {}).get("text", "") or "").strip()
                parts.append(f"[voice] {voice_text}".strip())
            elif item.get("type") == 4:
                file_name = (item.get("file_item") or {}).get("file_name", "unknown")
                parts.append(f"[file: {file_name}]")
            elif item.get("type") == 5:
                parts.append("[video]")
        text = "\n".join(part for part in parts if part).strip()
        if not text:
            return None
        return InboundMessage(
            text=text,
            sender_id=user_id,
            channel=self.name,
            account_id=self.account_id,
            peer_id=user_id,
            media=media,
            raw=msg,
        )

    def receive(self) -> InboundMessage | None:
        if not self._token and not self.login():
            time.sleep(5)
            return None
        data = self._api_post(
            "ilink/bot/getupdates",
            {"get_updates_buf": self._get_updates_buf, "base_info": WEIXIN_BASE_INFO},
        )
        ret = data.get("ret", 0)
        errcode = data.get("errcode", 0)
        if (ret not in (None, 0)) or (errcode not in (None, 0)):
            if errcode == -14 or ret == -14:
                print(f"{RED}[weixin] Session expired; run with WEIXIN_TOKEN or rescan QR code{RESET}")
                self._token = ""
                self._save_state()
                time.sleep(60)
                return None
            raise RuntimeError(f"Weixin getupdates failed: ret={ret} errcode={errcode} errmsg={data.get('errmsg', '')}")
        if data.get("longpolling_timeout_ms"):
            self.poll_timeout = max(int(data["longpolling_timeout_ms"]) // 1000, 5)
            self._http.timeout = httpx.Timeout(self.poll_timeout + 10, connect=30)
        if data.get("get_updates_buf"):
            self._get_updates_buf = data["get_updates_buf"]
            self._save_state()
        for msg in data.get("msgs", []) or []:
            inbound = self._parse_message(msg)
            if inbound is not None:
                return inbound
        return None

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        if not self._token:
            return False
        context_token = self._context_tokens.get(to, "")
        if not context_token:
            print(f"{RED}[weixin] No context_token for {to}; reply after receiving a message from this user{RESET}")
            return False
        chunks = [text[i:i + WEIXIN_MAX_MESSAGE_LEN] for i in range(0, len(text), WEIXIN_MAX_MESSAGE_LEN)] or [""]
        try:
            for chunk in chunks:
                msg = {
                    "from_user_id": "",
                    "to_user_id": to,
                    "client_id": f"dclaw-{uuid.uuid4().hex[:12]}",
                    "message_type": MESSAGE_TYPE_BOT,
                    "message_state": MESSAGE_STATE_FINISH,
                    "item_list": [{"type": ITEM_TEXT, "text_item": {"text": chunk}}],
                    "context_token": context_token,
                }
                data = self._api_post("ilink/bot/sendmessage", {"msg": msg, "base_info": WEIXIN_BASE_INFO})
                if data.get("errcode", 0) not in (None, 0):
                    print(f"{RED}[weixin] Send: {data.get('errmsg', '?')}{RESET}")
                    return False
            return True
        except Exception as exc:
            print(f"{RED}[weixin] Send: {exc}{RESET}")
            return False

    def close(self) -> None:
        self._closed = True
        self._save_state()
        self._http.close()

class ChannelManager:
    def __init__(self, notify: Callable[[str], None] | None = None) -> None:
        self.channels: dict[str, Channel] = {}
        self._notify = notify or _default_info
        self.accounts: list[ChannelAccount] = []

    def register(self, channel: Channel) -> None:
        self.channels[channel.name] = channel
        self._notify(f"[+] Channel registered: {channel.name}")

    def list_channels(self) -> list[str]:
        return list(self.channels.keys())

    def get(self, name: str) -> Channel | None:
        return self.channels.get(name)

    def close_all(self) -> None:
        for ch in self.channels.values():
            ch.close()
