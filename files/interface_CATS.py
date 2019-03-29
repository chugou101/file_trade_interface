# -*- coding:utf-8 -*-
from abc import ABCMeta, abstractmethod
import pandas as pd
from time import time
from os import urandom
from binascii import b2a_hex
from io import StringIO
from enum import Enum
import re
from contextlib import contextmanager


class OrderStatus(Enum):
    # 已报
    reported = 0
    # 订单未完成, 部分成交
    filled = 1
    # 订单完成, 全部成交, Order.filled 等于 Order.amount
    held = 2
    # 部分撤单
    ptcanceled = 3
    # 订单完成, 已撤销, 可能有成交, 需要看 Order.filled 字段
    canceled = 4
    # 订单完成, 交易所已拒绝, 可能有成交, 需要看 Order.filled 字段
    rejected = 5
    # 柜台未接受
    unaccepted = 6

    """
    # 订单新创建未委托，用于盘前/隔夜单，订单在开盘时变为 open 状态开始撮合
    new = 8
    # 订单未完成, 无任何成交
    open = 0
    # 订单取消中，只有实盘会出现，回测/模拟不会出现这个状态
    pending_cancel = 9
    # 无效的
    """


class Currency(Enum):
    # RMB
    rmb = 0
    # US dollar
    usd = 1
    # HK dollar
    hkd = 2


def get_code_exchange(code):
    '''
    上海证券交易所证券代码分配规则
    https://biz.sse.com.cn/cs/zhs/xxfw/flgz/rules/sserules/sseruler20090810a.pdf

    深圳证券交易所证券代码分配规则
    http://www.szse.cn/main/rule/bsywgz/39744233.shtml
    '''
    if isinstance(code, int):
        return code, 'XSHG' if code >= 500000 else 'XSHE'
    elif isinstance(code, str):
        code = code.upper()
        if code[-5:] in ('.XSHG', '.XSHE'):
            return code[:-4], code[-4:]

        suffix = None
        match = re.search(r'[0-9]{6}', code)

        if not match:
            raise ValueError("Unknown security: {}".format(code))

        number = match.group(0)
        if 'SH' in code:
            suffix = 'XSHG'
        elif 'SZ' in code:
            suffix = 'XSHE'

        if suffix is None:
            suffix = 'XSHG' if int(number) >= 500000 else 'XSHE'
        return number, suffix


def get_security(number, market):
    """
    将文件中分开的交易品种代码和市场重新拼接成聚宽使用的代码形式
    market=1 上海；market=2 深圳
    """
    if market == 1:
        security = str(number) + ".XSHG"
    elif market == 3:
        security = str(number) + ".XSHE"
    return security


class Order(object):
    """
    订单类，若回报中无相关域，保持为None
    订单中需要包含账户相关（类型etc.）信息吗
    """

    def __init__(self, status, add_time, is_buy, amount, filled, security,
                 order_id, price, avg_cost, side, action, origin_status, ref, raw):
        # 状态，一个OrderStatus值
        self.status = status
        # 订单添加时间, datetime对象
        self.add_time = add_time
        # 买卖方向，bool值
        self.is_buy = is_buy
        # 委托数量
        self.amount = amount
        #  已经成交的股票数量
        self.filled = filled
        # 标的代码
        self.security = security
        # 订单ID
        self.order_id = str(order_id)
        # 平均成交价格
        self.price = price
        # 等同于price
        self.avg_cost = avg_cost
        # 多/空， 'long'/'short'， 股票均为多
        self.side = side
        # 开/平， 'open'/'close'
        self.action = action

        # 把原始的订单给保存下来
        # 原始订单状态
        self.origin_status = origin_status
        # 扫单软件名称
        self.ref = ref
        # 原始的回报记录
        if isinstance(raw, pd.DataFrame):
            self.raw = raw.to_dict()
        else:
            self.raw = dict(raw)
        # 交易类型， "F" 成交，"EC" ETF资金划转


class Portfolio:
    # 账户信息， 若回报中无相关域，保持为None
    def __init__(self, acct_type=None, acct=None, currency=None, inout_cash=None, available_cash=None,
                 transferable_cash=None, locked_cash=None,
                 margin=None, locked_margin=None, positions=None, long_positions=None, short_positions=None,
                 total_value=None,
                 returns=None, starting_cash=None, positions_value=None, locked_cash_by_purchase=None,
                 locked_cash_by_redeem=None, locked_amount_by_redeem=None, ref=None, raw=None):
        # 账户类型
        self.acct_type = acct_type,
        # 交易账户
        self.acct = acct,
        # 币种 {0,1,2}
        self.currency = currency,
        # 累计出入金
        self.inout_cash = inout_cash,
        # 可用资金, 可用来购买证券的资金
        self.available_cash = available_cash,
        # 可取资金, 即可以提现的资金, 不包括今日卖出证券所得资金
        self.transferable_cash = transferable_cash,
        # 挂单锁住资金
        self.locked_cash = locked_cash,
        # 保证金，股票、基金保证金都为100%
        self.margin = margin,
        # 冻结保证金
        self.locked_margin = locked_margin,
        # 等同于 long_positions
        self.positions = long_positions,
        # 多单的仓位, 一个 dict, key 是证券代码, value 是Position对象
        self.long_positions = long_positions,
        # 空单的仓位, 一个 dict, key 是证券代码, value 是Position对象
        self.short_positions = short_positions,
        # 总的权益, 包括现金, 保证金, 仓位的总价值, 可用来计算收益
        self.total_value = total_value,
        # 总权益的累计收益
        self.returns = returns,
        # 初始资金, 现在等于 inout_cash
        self.starting_cash = starting_cash,
        # 持仓价值
        self.positions_value = positions_value,
        # 基金申购未完成所冻结的金额
        self.locked_cash_by_purchase = locked_cash_by_purchase,
        # 基金赎回未到账的金额
        self.locked_cash_by_redeem = locked_cash_by_redeem,
        # 基金赎回时，冻结的份额
        self.locked_amount_by_redeem = locked_amount_by_redeem,

        # 扫单软件名称
        self.ref = ref
        # 原始的回报记录
        if isinstance(raw, pd.DataFrame):
            self.raw = raw.to_dict()
        else:
            self.raw = dict(raw)


class Position:
    def __init__(self, future_trade_mark=None, prev_position=None, security=None, price=None, acc_avg_cost=None,
                 avg_cost=None, hold_cost=None, init_time=None, transact_time=None,
                 total_amount=None, closeable_amount=None, today_amount=None,
                 locked_amount=None, value=None, side=None, ref=None, raw=None):
        """
        增加对期货品种的标记：
        1.future_trade_mark 投机套保标记（期货交易的不同模式）
        2.prev_position 昨日持仓数
        """
        # 投机套保标记
        self.future_trade_mark = future_trade_mark,
        # 昨仓数
        self.prev_position = prev_position,
        # 标的代码
        self.security = security
        # 最新行情价格
        self.price = price
        # 累计开仓成本,计算方法如下 买入: (old_amount * old_avg_cost + trade_amount * trade_price + commission) / (old_amount + trade_amount) 卖出: (old_amount * old_avg_cost - trade_amount * trade_price + commission) / (old_amount - trade_amount) 说明：commission是本次买入或者卖出的手续费
        self.acc_avg_cost = acc_avg_cost
        # 开仓均价，买入标的的加权平均价, 计算方法是: (buy_volume1 * buy_price1 + buy_volume2 * buy_price2 + …) / (buy_volume1 + buy_volume2 + …) 每次买入后会调整avg_cost, 卖出时avg_cost不变. 这个值也会被用来计算浮动盈亏.
        self.avg_cost = avg_cost
        # 持仓成本，针对期货有效。
        self.hold_cost = hold_cost
        # 建仓时间，格式为 datetime.datetime
        self.init_time = init_time
        # 最后交易时间，格式为 datetime.datetime
        self.transact_time = transact_time
        # 总仓位, 但不包括挂单冻结仓位
        self.total_amount = total_amount
        # 可卖出的仓位 / 场外基金持有份额
        self.closeable_amount = closeable_amount
        # 今天开的仓位
        self.today_amount = today_amount
        # 挂单冻结仓位
        self.locked_amount = locked_amount
        # 标的价值，计算方法是: price * total_amount * multiplier, 其中股票、基金的multiplier为1，期货为相应的合约乘数
        self.value = value
        # 多/空，'long' or 'short'
        self.side = side
        # 仓位索引，subportfolio index
        self.pindex = 0

        # 扫单软件名称
        self.ref = ref
        # 原始的回报记录
        if isinstance(raw, pd.DataFrame):
            self.raw = raw.to_dict()
        else:
            self.raw = dict(raw)


class BaseOrder:
    pass


class ABCInterface:
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def portfolio(self):
        """
        账户与持仓信息
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def sync(self):
        """
        从文件同步状态
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def order(self, security, amount, style=None, side='long', **kwargs):
        """
        下单接口
        :param security: 标的代码
        :param amount:  委托数量
        :param style: 委托方式
        :param side: 多/空，暂时不生效
        :param kwargs: 其他底层参数
        :return: 委托ID
        """
        raise NotImplementedError

    @contextmanager
    @abstractmethod
    def bucket(self):
        """
        实现批量下单的contextmanager
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_open_orders(self):
        """
        查询未完成的订单
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_orders(self, order_id=None, security=None, status=None):
        raise NotImplementedError

    def get_trades(self):
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order):
        raise NotImplementedError

    def margincash_open(self, security, amount, style=None, **kwargs):
        raise NotImplementedError

    def margincash_close(self, security, amount, style=None, **kwargs):
        raise NotImplementedError

    def margincash_direct_refund(self, value, **kwargs):
        raise NotImplementedError

    def marginsec_open(self, security, amount, style=None, **kwargs):
        raise NotImplementedError

    def marginsec_close(self, security, amount, style=None, **kwargs):
        raise NotImplementedError

    def marginsec_direct_refund(self, security, amount, **kwargs):
        raise NotImplementedError


class CATSInterface(ABCInterface):
    Fields = {
        "order": ["inst_type", "client_id",
                  "acct_type", "acct", "order_no", "symbol", "tradeside",
                  "ord_qty", "ord_price", "ord_type", "ord_param"],
        "order_report": ["client_id", "ord_no", "ord_status", "acct_type",
                         "acct", "cats_acct", "symbol", "tradeside", "ord_qty",
                         "ord_px", "ord_type", "ord_param", "corr_type", "corr_id",
                         "filled_qty", "avg_px", "cxl_qty", "ord_time", "err_msg", "write_time"],
        "order_held": ["accttype", "account", "symbol", "tradeside", "orderno", "tradeno",
                       "fillqty", "fillprice", "trdtype", "time"],
        "fund_report": ["a_type", "acct_type", "acct", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
    }

    OrderStatusMapping = {
        "0": OrderStatus.reported,
        "1": OrderStatus.filled,
        "2": OrderStatus.held,
        "3": OrderStatus.ptcanceled,
        "4": OrderStatus.canceled,
        "5": OrderStatus.rejected,
        "6": OrderStatus.unaccepted,
    }

    OrderStatusReverseMapping = {
        OrderStatus.reported: "0",
        OrderStatus.filled: "1",
        OrderStatus.held: "2",
        OrderStatus.ptcanceled: "3",
        OrderStatus.canceled: "4",
        OrderStatus.rejected: "5",
        OrderStatus.unaccepted: "6",
    }
    acct_stock = {"0", "C", "HK0", "SHDDF0", "SHDDFC", "SZDDF0", "SZDDFC", "S0"}
    acct_future = {"A", "DHA", "DWA", "ESA", "G2A", "GFGA", "GJA", "GLA", "HTA", "HUTA", "HXA", "HZA", "JXA", "XZA",
                   "YAA", "ZDA", "SA"}
    acct_option = {"0B"}

    class LimitOrder(BaseOrder):
        def __init__(self, price, ord_param=""):
            self.price = price
            self.type = "0"
            self.order_param = ord_param

    class OptionOPALimitOrder(BaseOrder):
        # OPA 代表期权限价 FOK
        def __init__(self, price):
            self.price = price
            self.type = "OPA"

    class MarketOrder(BaseOrder):
        def __init__(self, type=None, ord_param=""):
            self.price = 0
            self.type = type
            self.order_param = ord_param

    def __init__(self, order_list_file, order_report_file,
                 fund_report_file, order_held_file):
        self.bucket_mod = False
        self.order_queue = []

        self.csv_files = {
            "order": order_list_file,
            "order_held": order_held_file,
            "order_report": order_report_file,
            "fund_report": fund_report_file,
        }

        self.file_cur_offset = {}

        for k, path in self.csv_files.items():
            with open(path, "a+") as f:
                f.seek(0, 2)
                self.file_cur_offset[k] = f.tell()

        self.df = {}
        for k in self.csv_files:
            self.df[k] = pd.DataFrame(columns=self.Fields[k])

        self.id_mapping = {}

    @classmethod
    def _get_order_status(cls, status):
        return cls.OrderStatusMapping[str(status)]

    def _gen_instruction(self, template, **kwargs):
        col_template = self.Fields.get(template, None)
        if not col_template:
            raise ValueError("Unsupported record template: {}".format(template))
        else:
            rec = pd.DataFrame(data=[kwargs], columns=col_template)
            return rec

    def _gen_order_from_report(self, ref_df, status=None):
        result = []
        for _, row in ref_df.iterrows():
            rec = row.to_dict()
            # not sure: is_buy,add_time,side
            result.append(Order(status=self._get_order_status(rec["ord_status"]),
                                add_time=pd.to_datetime(rec["ord_time"], unit="s"),
                                is_buy=str(rec["tradeside"]) in {"1", "A", "F"},
                                amount=rec["ord_qty"],
                                filled=rec["filled_qty"],
                                security=rec["symbol"],
                                order_id=rec["client_id"],
                                price=rec["avg_px"],
                                avg_cost=rec["avg_px"],
                                side="long", action=["close", "open"][str(rec["tradeside"]) in {"1", "A", "F"}],
                                origin_status=rec["ord_status"], ref="CATS", raw=rec))
        return result

    def _dump(self, template, instruction):
        instruction.to_csv(self.csv_files[template], encoding="utf-8", mode="a",
                           header=False, index=False)

    def _save_order_to_disk(self, rec):
        if not self.bucket_mod:
            rec = self._gen_instruction(template="order", **rec)
            self.df["order"] = pd.concat([self.df["order"], rec], ignore_index=False)
            self._dump("order", rec)
        else:
            self.order_queue.append(rec)

    def gen_order_id(self):
        return str(int(time())) + b2a_hex(urandom(4)).decode

    def order(self, security, amount, style=None, side='long', **kwargs):
        # code, exchange = get_code_exchange(security)
        """
        1.type or style?  如果传入不在定义范围内的order类型，则默认为市价单；否则传入定义范围内的order类型
		2.CATS supports shorting futures and options while JoinQuant doesn't
        """
        order_style = self.MarketOrder() if not issubclass(type, BaseOrder) else style
        order_id = self.gen_order_id()
        rec = {"inst_type": "O",
               "client_id": kwargs.get("client_id", ""),
               "acct_type": kwargs.get("acct_type", ""),
               "acct": kwargs.get("acct", ""),
               "order_no": order_id,
               "symbol": security,
               "tradeside": ["1", "2"][amount > 0] if not kwargs.get("tradeside]", False) else kwargs["tradeside"],
               "ord_qty": abs(int(amount)),
               "ord_price": order_style.price,
               "ord_type": order_style.type,
               "ord_param": kwargs.get("ord_param", "")}
        self._save_order_to_disk(rec)
        return order_id

    @contextmanager
    def bucket(self):
        # 进入批量委托模式
        self.bucket_mod = True

        yield

        # 处理委托
        recs = [self._gen_instruction(template="order", **rec) for rec in self.order_queue]
        recs = pd.concat(recs, ignore_index=False)
        self.df["order"] = pd.concat([self.df["order"], recs], ignore_index=False)
        self._dump("order", recs)

        # 提交委托
        self.order_queue = []
        self.bucket_mod = False

    def sync(self):
        for k, path in self.csv_files.items():
            with open(path, "a+") as f:
                # f.seek(0, 2)
                f.seek(0, 0)
                # if self.file_cur_offset[k] < f.tell():
                if True:
                    buffer = StringIO()
                    # f.seek(self.file_cur_offset[k])
                    buffer.write(f.read())
                    buffer.seek(0)
                    self.file_cur_offset[k] = f.tell()
                    df = pd.read_csv(buffer, names=self.Fields[k], memory_map=True, index_col=False)
                    self.df[k] = pd.concat([self.df[k], df])
        # 潜在bug，如果是数值ID会被解析成浮点数转string
        id_mapping = {str(i["client_id"]): str(i["order_no"]) for i in
                      self.df["order_report"][["client_id", "order_no"]].to_dict('records')}
        self.id_mapping.update(id_mapping)

    def get_open_orders(self):
        self.sync()
        df = self.df["order_report"]
        # unaccepted=unreported ??
        df = df[df["ord_status"].astype("str").isin({"0", "1", "3", "6"})].groupby(["symbol", "ord_no"]).tail(1)
        return self._gen_order_from_report(df.reset_index())

    def get_held_orders(self):
        self.sync()
        df = self.df["order_held"]
        result = []
        for _, row in df.iterrows():
            rec = row.to_dict()
            # not sure: is_buy,add_time,side
            result.append(Order(status=self._get_order_status(2),
                                add_time=pd.to_datetime(rec["time"], unit="ms"),
                                is_buy=str(rec["tradeside"]) in {"1", "A", "F"},
                                amount=rec["fillqty"],
                                filled=rec["fillqty"],
                                security=rec["symbol"],
                                order_id=rec["order_no"],
                                price=rec["fillprice"],
                                avg_cost=rec["fillprice"],
                                action=["close", "open"][str(rec["tradeside"]) in {"1", "A", "F"}],
                                ref="CATS", raw=rec))
        return result

    def get_orders(self, order_id=None, security=None, status=None):
        self.sync()
        df = self.df["order_report"]
        if order_id:
            df = df[df["client_id"] == order_id]
        if security:
            df = df[df["symbol"] == security]
        if status:
            df = df[df["ord_status"] == status]

        df = df.groupby(["symbol", "client_id"]).tail(1)
        test = self._gen_order_from_report(df.reset_index())
        return test

    def cancel_order(self, order):
        self.sync()
        order_no = self.id_mapping[order.order_id if isinstance(order, Order) else str(order)]
        order_id = self.gen_order_id()
        rec = {"inst_type": "C",
               "acct_type": order.acct_type,
               "acct": order.acct,
               "order_no": order_no}
        self._save_order_to_disk(rec)
        return order_id

    @property
    def position(self, acct_type=None, acct=None):
        """
        两种list(position&name)分别用于存储份额对象和证券代码
     	future_trade_mark=None,prev_position=None,security = None, price = None, acc_avg_cost = None,
                 avg_cost = None, hold_cost = None, init_time = None, transact_time = None,
                 total_amount = None, closeable_amount = None, today_amount = None,
                 locked_amount = None, value = None, side = None, ref=None, raw=None
        根据CATS说明，持仓方向对期货账户才有意义，是否意味着只有期货才有short position。但没有说明具体方向代表的值，
        目前将rec["s5"]==1看做空头
        """
        self.sync()
        df = self.df["fund_report"]
        df = df[df["a_type"] == "P"]
        if acct_type:
            df = df[df["acct_type"] == acct_type]
        if acct:
            df = df[df["acct"] == acct]
        shortpositions = []
        shortname = []
        longpositions = []
        longname = []
        for _, row in df.iterrows():
            rec = row.to_dict()
            if acct_type in self.acct_future and str(rec["s5"]) == "1":  # 空头持仓
                shortpositions.append(Position(security=rec["s1"], avg_cost=rec["s4"],
                                               total_amount=rec["s2"], closeable_amount=rec["s3"],
                                               value=rec["s8"], side="short", ref="CATS", raw=rec,
                                               future_trade_mark=rec["s6"] if acct_type in self.acct_future else None,
                                               prev_position=rec["s7"] if acct_type in self.acct_future else None))
                shortname.append(rec["s1"])
            else:
                longpositions.append(Position(security=rec["s1"], avg_cost=rec["s4"],
                                              total_amount=rec["s2"], closeable_amount=rec["s3"],
                                              value=rec["s8"], side="short", ref="CATS", raw=rec,
                                              future_trade_mark=rec["s6"] if acct_type in self.acct_future else None,
                                              prev_position=rec["s7"] if acct_type in self.acct_future else None))
                longname.append(rec["s1"])
            return dict(zip(longname, longpositions)), dict(zip(shortname, shortpositions))

    @property
    def portfolio(self):
        """
        目前认为对不同账户类型的不同交易账户是一个portfolio
        acct_type=None,acct=None,currency=None,inout_cash=None, available_cash=None, transferable_cash=None, locked_cash=None,
                 margin=None,locked_margin=None, positions=None, long_positions=None, short_positions=None, total_value=None,
                 returns=None, starting_cash=None, positions_value=None, locked_cash_by_purchase=None,
                 locked_cash_by_redeem=None, locked_amount_by_redeem=None, ref=None, raw=None
        """
        self.sync()
        #    df = df[df["ord_status"].astype("str").isin({"0","1","3","6"})].groupby(["symbol", "ord_no"]).tail(1)
        result = []
        df = self.df["fund_report"]
        df = df[df["a_type"] == "F"].reset_index()
        for _, row in df.iterrows():
            rec = row.to_dict()
            positions, shortpositions = self.position(acct_type=rec["acct_type"], acct=rec["acct"])
            longpositions = positions
            result.append(Portfolio(acct_type=rec["acct_type"], acct=rec["acct"], currency=rec["s2"],
                                    total_value=rec["s3"], available_cash=rec["s4"],
                                    margin=rec["s5"] if str(rec["acct_type"]) in self.acct_future else None,
                                    locked_margin=rec["s6"] if str(rec["acct_type"]) in self.acct_future else None,
                                    positions=positions, short_positions=shortpositions, long_positions=longpositions,
                                    ref="CATS", raw=rec))

        return result

    # 融资买入
    def margincash_open(self, security, amount, style=None, **kwargs):
        return self.order(security=security, amount=amount, style=style,
                          tradeside="A", **kwargs)

    # 卖券还款
    def margincash_close(self, security, amount, style=None, **kwargs):
        return self.order(security=security, amount=amount, style=style,
                          tradeside="D", **kwargs)

    # 融券卖出
    def marginsec_open(self, security, amount, style=None, **kwargs):
        return self.order(security=security, amount=amount, style=style,
                          tradeside="B", **kwargs)

    # 买券还券
    def marginsec_close(self, security, amount, style=None, **kwargs):
        return self.order(security=security, amount=amount, style=style,
                          tradeside="C", **kwargs)

    """
    ETF申购赎回和期货期权的下单函数还没有写
    """
    """
    def margincash_direct_refund(self, value, **kwargs):
        order_id = self.gen_order_id()
        rec = {"InstructionType": "O",
               "UserRequestID": order_id,
               "UserID": "",
               "BAMapID": kwargs.get("BAMapID", ""),
               "PCID": "",
               "IssueCode": "",
               "BuySell": "G",
               "OpenClose": "",
               "Quantity": value,
               "Price": "",
               "OrderType": "",
               "Option": ""}
        self._save_order_to_disk(rec)
        return order_id
    
    def marginsec_direct_refund(self, value, **kwargs):
        order_id = self.gen_order_id()
        rec = {"InstructionType": "O",
               "UserRequestID": order_id,
               "UserID": "",
               "BAMapID": kwargs.get("BAMapID", ""),
               "PCID": "",
               "IssueCode": "",
               "BuySell": "I",
               "OpenClose": "",
               "Quantity": value,
               "Price": "",
               "OrderType": "",
               "Option": ""}
        self._save_order_to_disk(rec)
        return order_id
	"""


# ABCInterface.register(DTSInterface)

if __name__ == "__main__":
    t = CATSInterface(order_list_file="./OrderList.csv",
                      order_report_file="./OrderReportList.csv",
                      order_held_file="./OrderHeldFile",
                      # query_order_file="./QueryOrderList.csv",
                      fund_report_file="./FundReport.csv"
                      )
    t.get_orders()
