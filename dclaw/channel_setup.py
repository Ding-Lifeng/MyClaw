from __future__ import annotations

from .channels import (
    HAS_HTTPX,
    ChannelAccount,
    ChannelManager,
    FeishuChannel,
    WeixinPersonalChannel,
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

    if config.channels.weixin_enabled and HAS_HTTPX:
        wx_personal_acc = ChannelAccount(
            channel="weixin",
            account_id="weixin-primary",
            config={
                "allow_from": config.channels.weixin_allow_from,
                "base_url": config.channels.weixin_base_url,
                "route_tag": config.channels.weixin_route_tag,
                "token": config.channels.weixin_token,
                "state_dir": config.channels.weixin_state_dir,
                "poll_timeout": config.channels.weixin_poll_timeout,
            },
        )
        mgr.accounts.append(wx_personal_acc)
        mgr.register(WeixinPersonalChannel(wx_personal_acc))
