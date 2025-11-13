from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProfitLossInput:
    buy_price: float
    sell_price: float
    quantity: int
    fee_rate: float = 0.00015
    tax_rate: float = 0.0025


def calculate_profit_loss(params: ProfitLossInput) -> dict[str, float]:
    gross_buy = params.buy_price * params.quantity
    gross_sell = params.sell_price * params.quantity
    fees = (gross_buy + gross_sell) * params.fee_rate
    taxes = gross_sell * params.tax_rate
    net_profit = gross_sell - gross_buy - fees - taxes
    profit_rate = net_profit / gross_buy if gross_buy else 0.0
    return {
        "gross_buy": gross_buy,
        "gross_sell": gross_sell,
        "fees": fees,
        "taxes": taxes,
        "net_profit": net_profit,
        "profit_rate": profit_rate,
    }


def calculate_average_price(total_cost: float, total_shares: int) -> float:
    if total_shares == 0:
        return 0.0
    return total_cost / total_shares


def apply_fx(amount: float, rate: float, invert: bool = False) -> float:
    """
    단순 환율 계산 유틸리티.
    - rate: 1단위 외화당 원화 가격 (예: 1 USD = 1457.0 KRW)
    - invert=True: 원화 -> 외화 (amount KRW / rate)
    - invert=False: 외화 -> 원화 (amount * rate)
    """
    if rate == 0:
        return 0.0
    return amount / rate if invert else amount * rate


