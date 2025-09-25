#!/usr/bin/env python3
"""
Common utilities for all trainers
"""

import datetime
import torch
import numpy as np
import random

def GetPriceList(content, name_num=0):
    """Extract price list from data content"""
    price_list = []
    name_list = ['BTCBitcoin_price', 'ETHEthereum_price']
    desired = name_list[name_num]
    cnt = 0
    for name in content:
        if desired in name:
            if cnt == 0:
                start_date = name[0:8]
                cnt += 1
            price_list.append(content[name])
    return price_list, start_date

def dateAdd(date, interval=1):
    """Add days to date string"""
    dt = datetime.datetime.strptime(date, "%Y%m%d")
    dt = dt + datetime.timedelta(interval)
    date1 = dt.strftime("%Y%m%d")
    return date1

def seed_torch(seed):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
