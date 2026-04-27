from __future__ import annotations

from .channels import (
    HAS_HTTPX,
    ChannelAccount,
    ChannelManager,
    FeishuChannel,
    WechatChannel,
)
from .config import AppConfig


def register_configured_channels(config: AppConfig, mgr: ChannelManager) -> None:
    fs_id = config.channels.feishu_app_id
    fs_secret = config.channels.feishu_app_secret
    if fs_id and fs_secret and HAS_HTTPX:
        fs_acc = ChannelAccount(
            channel="feishu",
            account_id="feishu-primary",
            config={
                "app_id": fs_id,
                "app_secret": fs_secret,
                "encrypt_key": config.channels.feishu_encrypt_key,
                "bot_open_id": config.channels.feishu_bot_open_id,
                "is_lark": config.channels.feishu_is_lark,
            },
        )
        mgr.accounts.append(fs_acc)
        mgr.register(FeishuChannel(fs_acc))

    wx_webhook = config.channels.wechat_webhook_url
    if wx_webhook and HAS_HTTPX:
        wx_acc = ChannelAccount(
            channel="wechat",
            account_id="wechat-primary",
            config={"webhook_url": wx_webhook},
        )
        mgr.accounts.append(wx_acc)
        mgr.register(WechatChannel(wx_acc))
