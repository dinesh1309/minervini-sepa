"""
Discord Alert Engine for SEPA Trading Scanner

Sends trading alerts to Discord via webhook.
Alert types:
- Market condition change (uptrend/downtrend/rally attempt)
- FTD detected
- Stock crosses pivot level
- Stock hits danger level (below SMA 50)
- Scan summary
"""

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_webhook_url() -> Optional[str]:
    """Get Discord webhook URL from environment."""
    url = os.environ.get('DISCORD_WEBHOOK_URL')
    if not url:
        # Try loading from .env file
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith('DISCORD_WEBHOOK_URL='):
                        url = line.strip().split('=', 1)[1]
                        break
    return url


def send_discord_message(content: str, embeds: List[Dict] = None) -> bool:
    """
    Send a message to Discord via webhook.

    Args:
        content: Text content (max 2000 chars)
        embeds: Optional list of embed objects for rich formatting

    Returns:
        True if sent successfully
    """
    url = get_webhook_url()
    if not url:
        logger.warning("DISCORD_WEBHOOK_URL not set. Alerts disabled.")
        return False

    payload = {}
    if content:
        payload["content"] = content[:2000]
    if embeds:
        payload["embeds"] = embeds[:10]  # Discord max 10 embeds

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 204:
            logger.info("Discord alert sent successfully")
            return True
        else:
            logger.error(f"Discord webhook failed: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Discord webhook error: {e}")
        return False


def build_market_condition_embed(index_result: Dict) -> Dict:
    """Build a Discord embed for market condition."""
    condition = index_result.get("condition_raw", index_result.get("condition", "UNKNOWN"))
    name = index_result.get("name", "Index")

    color_map = {
        "CONFIRMED_UPTREND": 0x00FF00,   # Green
        "UNDER_PRESSURE": 0xFFA500,       # Orange
        "RALLY_ATTEMPT": 0xFFFF00,        # Yellow
        "DOWNTREND": 0xFF0000,            # Red
    }
    color = color_map.get(condition, 0x808080)

    icon_map = {
        "CONFIRMED_UPTREND": "🟢",
        "UNDER_PRESSURE": "🟡",
        "RALLY_ATTEMPT": "🔵",
        "DOWNTREND": "🔴",
    }
    icon = icon_map.get(condition, "⚪")

    fields = [
        {"name": "Price", "value": str(index_result.get("price", "—")), "inline": True},
        {"name": "SMA 50", "value": str(index_result.get("sma50", "—")), "inline": True},
        {"name": "SMA 200", "value": str(index_result.get("sma200", "—")), "inline": True},
        {"name": "Dist Days", "value": str(index_result.get("distribution_days", "—")), "inline": True},
    ]

    if index_result.get("rally_attempt_day"):
        fields.append({"name": "Rally Day", "value": str(index_result["rally_attempt_day"]), "inline": True})

    if index_result.get("ftd_detected"):
        fields.append({"name": "🎯 FTD", "value": f"DETECTED on {index_result.get('ftd_date', '?')}", "inline": False})

    return {
        "title": f"{icon} {name} — {condition.replace('_', ' ')}",
        "description": index_result.get("details", ""),
        "color": color,
        "fields": fields,
        "footer": {"text": f"Data as of {index_result.get('as_of', '?')}"},
    }


def build_stock_alert_embed(stock: Dict, alert_type: str) -> Dict:
    """Build a Discord embed for a stock alert."""
    name = stock.get("name", stock.get("symbol", "?"))
    price = stock.get("price", 0)
    pivot = stock.get("pivot")

    if alert_type == "near_pivot":
        dist = stock.get("dist_from_pivot", 0)
        return {
            "title": f"📍 {name} — Near Pivot ({dist}%)",
            "description": f"Price: {price} | Pivot: {pivot} | Distance: {dist}%",
            "color": 0x00BFFF,  # Cyan
            "fields": [
                {"name": "Trend Template", "value": stock.get("trend_template", "—"), "inline": True},
                {"name": "vs SMA 50", "value": stock.get("price_vs_sma50", "—"), "inline": True},
            ],
        }
    elif alert_type == "tt_pass":
        return {
            "title": f"✅ {name} — Trend Template PASS",
            "description": f"Price: {price} | TT: {stock.get('trend_template', '—')}",
            "color": 0x00FF00,
            "fields": [
                {"name": "Pivot", "value": str(pivot) if pivot else "—", "inline": True},
                {"name": "Distance", "value": f"{stock.get('dist_from_pivot', '—')}%", "inline": True},
                {"name": "vs SMA 50", "value": stock.get("price_vs_sma50", "—"), "inline": True},
            ],
        }
    elif alert_type == "danger":
        return {
            "title": f"⚠️ {name} — Below SMA 50",
            "description": f"Price: {price} | SMA 50: {stock.get('sma50', '—')}",
            "color": 0xFF4500,  # Red-orange
        }
    elif alert_type == "vcp":
        v = stock.get("vcp", {})
        return {
            "title": f"🔍 {name} — VCP Pattern Detected",
            "description": f"Score: {v.get('score', 0)}/100 | Contractions: {v.get('contractions', 0)}",
            "color": 0x9B59B6,  # Purple
            "fields": [
                {"name": "Depths", "value": " > ".join(f"{d}%" for d in v.get("depths", [])), "inline": False},
                {"name": "Detected Pivot", "value": str(v.get("detected_pivot", "—")), "inline": True},
                {"name": "Volume Dry-up", "value": "YES" if v.get("volume_dry_up") else "NO", "inline": True},
            ],
        }

    return {"title": f"{name}", "description": alert_type, "color": 0x808080}


def generate_alerts(index_results: List[Dict], stock_results: List[Dict]) -> List[Dict]:
    """
    Analyze scan results and generate alerts.

    Returns list of alert dicts with type and embed.
    """
    alerts = []

    # Market condition alerts (always send)
    for idx in index_results:
        if "error" in idx:
            continue
        alerts.append({
            "type": "market_condition",
            "embed": build_market_condition_embed(idx),
            "priority": "high" if idx.get("ftd_detected") else "normal",
        })

    # Stock alerts
    for stock in stock_results:
        if "error" in stock:
            continue

        # Alert: TT passes
        if stock.get("tt_pass"):
            alerts.append({
                "type": "tt_pass",
                "embed": build_stock_alert_embed(stock, "tt_pass"),
                "priority": "high",
            })

        # Alert: Near pivot (within 5%)
        dist = stock.get("dist_from_pivot")
        if dist is not None and 0 <= dist <= 5:
            alerts.append({
                "type": "near_pivot",
                "embed": build_stock_alert_embed(stock, "near_pivot"),
                "priority": "high",
            })

        # Alert: VCP detected
        vcp = stock.get("vcp", {})
        if vcp.get("valid"):
            alerts.append({
                "type": "vcp",
                "embed": build_stock_alert_embed(stock, "vcp"),
                "priority": "medium",
            })

    return alerts


def send_scan_alerts(index_results: List[Dict], stock_results: List[Dict], timestamp: str) -> int:
    """
    Generate and send all alerts from a scan.

    Args:
        index_results: Index scan results
        stock_results: Stock scan results
        timestamp: Scan timestamp

    Returns:
        Number of alerts sent
    """
    alerts = generate_alerts(index_results, stock_results)

    if not alerts:
        logger.info("No alerts to send")
        return 0

    # Send header
    date_str = timestamp[:10]
    header = f"**📊 SEPA Watchlist Scan — {date_str}**"
    send_discord_message(header)

    # Send market condition embeds first
    market_embeds = [a["embed"] for a in alerts if a["type"] == "market_condition"]
    if market_embeds:
        send_discord_message("", embeds=market_embeds)

    # Send stock alerts
    stock_embeds = [a["embed"] for a in alerts if a["type"] != "market_condition"]
    if stock_embeds:
        # Discord allows max 10 embeds per message
        for i in range(0, len(stock_embeds), 10):
            batch = stock_embeds[i:i+10]
            send_discord_message("", embeds=batch)
    elif not stock_embeds:
        # No stock alerts — send summary
        tt_pass = sum(1 for s in stock_results if s.get("tt_pass") and "error" not in s)
        total = sum(1 for s in stock_results if "error" not in s)
        send_discord_message(f"📋 {tt_pass}/{total} stocks pass Trend Template. No actionable alerts.")

    sent = len(alerts)
    logger.info(f"Sent {sent} alerts to Discord")
    return sent
