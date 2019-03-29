# -*- coding:utf-8 -*-
from abc import ABCMeta, abstractmethod
import pandas as pd
from time import time, sleep
from os import urandom
from binascii import b2a_hex
from io import StringIO
from enum import Enum
import re
from contextlib import contextmanager


class OrderStatus(Enum):
    # 订单新创建未委托，用于盘前/隔夜单，订单在开盘时变为 open 状态开始撮合
    new = 8
    # 订单未完成, 无任何成交
    open = 0
    # 订单未完成, 部分成交
    filled = 1
    # 订单完成, 已撤销, 可能有成交, 需要看 Order.filled 字段
    canceled = 2
    # 订单完成, 交易所已拒绝, 可能有成交, 需要看 Order.filled 字段
    rejected = 3
    # 订单完成, 全部成交, Order.filled 等于 Order.amount
    held = 4
    # 订单取消中，只有实盘会出现，回测/模拟不会出现这个状态
    pending_cancel = 9
    # 无效的


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
    code = str(number)
    if len(code) < 6:
        code = "0" * (6 - len(code)) + code
    if market == 1:
        return code + ".XSHG"
    elif market == 2:
        return code + ".XSHE"


class Order(object):
    """
    订单类，若回报中无相关域，保持为None
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


class Portfolio:
    # 账户信息， 若回报中无相关域，保持为None
    def __init__(self, inout_cash=None, available_cash=None, transferable_cash=None, locked_cash=None,
                 margin=None, positions=None, long_positions=None, short_positions=None, total_value=None,
                 returns=None, starting_cash=None, positions_value=None, locked_cash_by_purchase=None,
                 locked_cash_by_redeem=None, locked_amount_by_redeem=None, ref=None, raw=None):
        # 累计出入金
        self.inout_cash = inout_cash if inout_cash else 0
        # 可用资金, 可用来购买证券的资金
        self.available_cash = available_cash if available_cash else 0
        # 可取资金, 即可以提现的资金, 不包括今日卖出证券所得资金
        self.transferable_cash = transferable_cash if transferable_cash else 0
        # 挂单锁住资金
        self.locked_cash = locked_cash if locked_cash else 0
        # 保证金，股票、基金保证金都为100%
        self.margin = margin
        # 等同于 long_positions
        self.positions = long_positions if long_positions else {}
        # 多单的仓位, 一个 dict, key 是证券代码, value 是Position对象
        self.long_positions = long_positions if long_positions else {}
        # 空单的仓位, 一个 dict, key 是证券代码, value 是Position对象
        self.short_positions = short_positions if short_positions else {}
        # 总的权益, 包括现金, 保证金, 仓位的总价值, 可用来计算收益
        self.total_value = total_value if total_value else 0
        # 总权益的累计收益
        self.returns = returns if returns else 0
        # 初始资金, 现在等于 inout_cash
        self.starting_cash = starting_cash if starting_cash else 0
        # 持仓价值
        self.positions_value = positions_value if positions_value else 0
        # 基金申购未完成所冻结的金额
        self.locked_cash_by_purchase = locked_cash_by_purchase if locked_cash_by_purchase else 0
        # 基金赎回未到账的金额
        self.locked_cash_by_redeem = locked_cash_by_redeem if locked_cash_by_redeem else 0
        # 基金赎回时，冻结的份额
        self.locked_amount_by_redeem = locked_amount_by_redeem if locked_amount_by_redeem else 0

        # 扫单软件名称
        self.ref = ref
        # 原始的回报记录
        if isinstance(raw, pd.DataFrame):
            self.raw = raw.to_dict()
        else:
            self.raw = dict(raw)


class Position:
    def __init__(self, security=None, price=None, acc_avg_cost=None,
                 avg_cost=None, hold_cost=None, init_time=None, transact_time=None,
                 total_amount=None, closeable_amount=None, today_amount=None,
                 locked_amount=None, value=None, side=None, ref=None, raw=None):
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
    def sync(self, target=None):
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


class DTSInterface(ABCInterface):
    Fields = {
        "order": ["InstructionType", "UserRequestID",
                  "UserID", "BAMapID", "PCID", "IssueCode", "BuySell",
                  "OpenClose", "Quantity", "Price", "OrderType", "Option"],
        "order_report": ["UserRequestID", "UserID", "BAMapID", "IssueCode",
                         "MarketCode", "PostID", "BuySell", "OpenClose", "Quantity", "Price", "OrderTime",
                         "OrderAcceptNo", "OrderStatus", "WorkingQty", "AlExecutionQty",
                         "AlExecutionValue", "ExecutionQty", "ExecutionValue", "ExecutionPrice",
                         "ExecutionTime", "ExecutionNo", "PCID", "RejectReason"],
        "query_order": ["InstructionType", "UserID", "BAMapID", "InvestorID", "Option"],
        "fund_report": ["BAMapID", "AssertValue", "AvlFund", "ValutionPL", "OrderFund",
                        "Amount"],
        "position_report": ["PosID", "BAMapID", "IssueCode", "IssueName", "LastPrice",
                            "MarketCode", "Quantity", "AvlQuantity", "BS", "AvgPrice", "CostValue",
                            "PL", "Fare", "AvlFund", "AssertValue", "ValutionPLM2D", "_CostValue",
                            "CumulationPL", "CumulationFare"]
    }

    OrderStatusMapping = {
        "0": OrderStatus.open,
        "1": OrderStatus.filled,
        "2": OrderStatus.held,
        "4": OrderStatus.canceled,
        "7": OrderStatus.new,
        "8": OrderStatus.rejected,
    }

    class LimitOrder(BaseOrder):
        def __init__(self, price):
            self.price = price
            self.type = "0"

    class MarketOrder(BaseOrder):
        def __init__(self, type=None):
            self.price = 0
            if type is None:
                self._type = {"XSHE": "H",
                              "XSHG": "X"}
                self.type = None
            else:
                # 支持文档中记载的其他相当方式，直接传入对应选项
                self.type = type

    def __init__(self, order_list_file, order_report_file, query_order_file,
                 fund_report_file, position_report_file):
        self.bucket_mod = False
        self.order_queue = []

        self.csv_files = {
            "order": order_list_file,
            "order_report": order_report_file,
            "query_order": query_order_file,
            "fund_report": fund_report_file,
            "position_report": position_report_file
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

    def _gen_order_from_report(self, ref_df):
        result = []
        for _, row in ref_df.iterrows():
            rec = row.to_dict()
            result.append(Order(status=self._get_order_status(rec["OrderStatus"]),
                                add_time=pd.to_datetime(rec["OrderTime"], unit="s"),
                                is_buy=str(rec["BuySell"]) in {"3", "A"},
                                amount=rec["Quantity"],
                                filled=rec["AlExecutionQty"],
                                security=".".join(get_code_exchange(str(rec["IssueCode"]))),
                                order_id=rec["UserRequestID"],
                                price=rec["ExecutionPrice"],
                                avg_cost=rec["ExecutionPrice"],
                                side="long", action=["close", "open"][str(rec["BuySell"]) in {"3", "A"}],
                                origin_status=rec["OrderStatus"], ref="DTS", raw=rec))
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

    def _save_query_to_disk(self, rec):
        rec = self._gen_instruction(template="query_order", **rec)
        self.df["query"] = pd.concat([self.df["query_order"], rec], ignore_index=False)
        self._dump("query_order", rec)

    def gen_order_id(self):
        return str(int(time())) + b2a_hex(urandom(4)).decode()

    def order(self, security, amount, style=None, side='long', **kwargs):
        code, exchange = get_code_exchange(security)
        order_style = self.MarketOrder() if not issubclass(type, BaseOrder) else style
        order_id = self.gen_order_id()
        rec = {"InstructionType": "O",
               "UserRequestID": order_id,
               "UserID": "",
               "BAMapID": kwargs.get("BAMapID", ""),
               "PCID": "",
               "IssueCode": code,
               "BuySell": ["1", "3"][amount > 0] if not kwargs.get("BuySell", False) else kwargs["BuySell"],
               "OpenClose": "",
               "Quantity": abs(int(amount)),
               "Price": order_style.price,
               "OrderType": order_style.type if order_style.type else order_style._type[exchange],
               "Option": ""}
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

    def sync(self, target=None):
        for k, path in self.csv_files.items() if target is None else [(target, self.csv_files[target])]:
            with open(path, "a+") as f:
                f.seek(0, 2)
                # f.seek(0, 0)
                if self.file_cur_offset[k] < f.tell():
                # if True:
                    buffer = StringIO()
                    f.seek(self.file_cur_offset[k])
                    buffer.write(f.read())
                    buffer.seek(0)
                    self.file_cur_offset[k] = f.tell()
                    df = pd.read_csv(buffer, names=self.Fields[k], memory_map=True, index_col=False)
                    self.df[k] = pd.concat([self.df[k], df])

        # 潜在bug，如果是数值ID会被解析成浮点数转string
        id_mapping = {str(i["UserRequestID"]): str(i["PCID"]) for i in
                      self.df["order_report"][["UserRequestID", "PCID"]].to_dict('records')}
        self.id_mapping.update(id_mapping)

    def get_open_orders(self):
        self.sync(target="order_report")
        df = self.df["order_report"]
        df = df[df["OrderStatus"].astype("str").isin({"0", "1", "7"})].groupby(["IssueCode", "OrderAcceptNo"]).tail(1)
        return self._gen_order_from_report(df.reset_index())

    def get_orders(self, order_id=None, security=None, status=None):
        self.sync(target="order_report")
        df = self.df["order_report"]
        if order_id:
            df = df[df["UserRequestID"] == order_id]
        if security:
            df = df[df["IssueCode"] == security]
        if status:
            df = df[df["OrderStatus"] == status]

        df = df.groupby(["IssueCode", "OrderAcceptNo"]).tail(1)
        test = self._gen_order_from_report(df.reset_index())
        return test

    def cancel_order(self, order):
        self.sync(target="order_report")
        pcid = self.id_mapping[order.order_id if isinstance(order, Order) else str(order)]
        order_id = self.gen_order_id()
        rec = {"InstructionType": "C",
               "UserRequestID": order_id,
               "PCID": pcid}

        self._save_order_to_disk(rec)
        return order_id

    def query_order(self, query, **kwargs):
        """
        "InstructionType", "UserID", "BAMapID", "InvestorID", "Option"
        """
        rec = {"InstructionType": query,
               "UserID": kwargs.get("UserID", ""),
               "BAMapID": kwargs.get("BAMapID", ""),
               "InvestorID": kwargs.get("InvestorID", ""),
               "Option": kwargs.get("Option", "")
               }
        self._save_query_to_disk(rec)

    @property
    def position(self):
        """
        两种list(position&name)分别用于存储份额对象和证券代码
        目前用于区分不同份额的是：证券代码+多/空头
        Q:
        账号问题 加不加BAMapID
        简单groupby如何解决持仓某只股票后又清仓的问题
        代码是否要返回到 xxxx.xxxx的形式
        """
        self.query_order("P")
        self.sync("position_report")
        df = self.df["position_report"].groupby(["PosID", "IssueCode"]).tail(1)
        shortpositions = []
        shortname = []
        longpositions = []
        longname = []
        for _, row in df.iterrows():
            rec = row.to_dict()
            securitycode = get_security(rec["IssueCode"], rec["MarketCode"])
            if rec["PosID"] == 1:  # 空头持仓
                shortpositions.append(Position(security=securitycode,
                                               price=rec["LastPrice"],
                                               avg_cost=rec["AvgPrice"],
                                               total_amount=rec["Quantity"],
                                               closeable_amount=rec["AvlQuantity"],
                                               value=rec["AssertValue"],
                                               side="short",
                                               ref="DTS",
                                               raw=rec))
                shortname.append(securitycode)
            elif rec["PosID"] == 3:
                longpositions.append(Position(security=securitycode,
                                              price=rec["LastPrice"],
                                              avg_cost=rec["AvgPrice"],
                                              total_amount=rec["Quantity"],
                                              closeable_amount=rec["AvlQuantity"],
                                              value=rec["AssertValue"],
                                              side="long",
                                              ref="DTS",
                                              raw=rec))
                longname.append(securitycode)
        return dict(zip(longname, longpositions)), dict(zip(shortname, shortpositions))

    @property
    def portfolio(self):
        """
        取portfolio信息时默认采用最新一条记录（文件最后一条），但这里没有指定理财账户，可能会有问题
        inout_cash=None, available_cash=None, transferable_cash=None, locked_cash=None,
                 margin=None, positions=None, long_positions=None, short_positions=None, total_value=None,
                 returns=None, starting_cash=None, positions_value=None, locked_cash_by_purchase=None,
                 locked_cash_by_redeem=None, locked_amount_by_redeem=None, ref=None, raw=None
        """
        self.query_order("F")
        self.sync("fund_report")
        df = self.df["fund_report"].tail(1).reset_index()
        if not df.empty:
            rec = df.iloc[0].to_dict()
            avlcash = rec["AvlFund"]  # 可用资金
            lockedcash = rec["OrderFund"]  # 冻结资金
            margin = rec["AssertValue"]
            # property不用调用，直接当作属性请求就行
            positions, shortpositions = self.position
            longpositions = positions
            totalvalue = rec["AssertValue"]
            cost = rec["Amount"]  # 买入成本
            returns = rec["ValutionPL"] / cost
            ref = "DTS"
        else:
            rec = {}
            avlcash = 0  # 可用资金
            lockedcash = 0  # 冻结资金
            margin = 0
            # property不用调用，直接当作属性请求就行
            positions, shortpositions = self.position
            longpositions = positions
            totalvalue = 0
            cost = 0  # 买入成本
            returns = 0
            ref = "DTS"
        p = Portfolio(available_cash=avlcash, locked_cash=lockedcash, margin=margin,
                      positions=positions, long_positions=longpositions,
                      total_value=totalvalue, returns=returns, ref=ref, raw=rec)

        return p

    def margincash_open(self, security, amount, style=None, **kwargs):
        return self.order(security=security, amount=amount, style=style,
                          BuySell="A", **kwargs)

    def margincash_close(self, security, amount, style=None, **kwargs):
        return self.order(security=security, amount=amount, style=style,
                          BuySell="B", **kwargs)

    def marginsec_open(self, security, amount, style=None, **kwargs):
        return self.order(security=security, amount=amount, style=style,
                          BuySell="C", **kwargs)

    def marginsec_close(self, security, amount, style=None, **kwargs):
        return self.order(security=security, amount=amount, style=style,
                          BuySell="D", **kwargs)

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


if __name__ == "__main__":
    # 示例
    interface = DTSInterface(order_list_file="Order.csv", 
                             order_report_file="OrderReport.csv",
                             query_order_file="QueryOrder.csv",
                             fund_report_file="FundReport.csv",
                             position_report_file="PositionReport.csv")
    # 下单
    order_id = interface.order(security="600660.XSHG", amount=100, style=DTSInterface.LimitOrder(price=23.33), side='long')

    # 批量下单，一次性写入文件
    with interface.bucket():
        for i in range(0,10):
            order_id = interface.order(security="600660.XSHG", amount=100, style=DTSInterface.LimitOrder(price=23.33), side='long')

    # 获取账户信息
    interface.portfolio

    # 获取持仓信息
    interface.position


