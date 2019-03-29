# -*- coding:utf-8 -*-
from abc import ABCMeta, abstractmethod
import pandas as pd
from time import time
from os import urandom,rename
from binascii import b2a_hex
from io import StringIO
from enum import Enum
import re
from contextlib import contextmanager
import datetime as dt

class OrderStatus(Enum):
    #撤单命令已发送
    CancelSending=0
    #撤单被拒绝
    CancelReject=1
    #撤单对应订单无法找到
    OrderNotFound=2
    #已撤单
    Canceled=3
    #待撤
    Cancelpending=4
    #过期
    Expired=5
    #成交
    Filled=6
    #部分成交，剩余挂单
    Partiallyfilled=7
    #部分成交，剩余已撤
    Partiallyfilledurout=8
    #队列中
    Queued=9
    #已接收
    Received=10
    #拒绝
    Rejected=11
    #报单失败
    Sendfailed=12
    #正在报送
    Sending=13
    #已报送
    Sent=14
    #未报送
    unsent=15

class Currency(Enum):
	#RMB
	rmb=0;
	#US dollar
	usd=1;
	#HK dollar
	hkd=2;
		
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

def get_symbol_exchange(code):
#返回 symbol+"SH"/"SZ"
    if isinstance(code, int):
        return code, 'SH' if code >= 500000 else 'SZ'
    elif isinstance(code, str):
        code = code.upper()
        if code[-3:] in ('.SH', '.SZ'):
            return code[:-3], code[-2:]
    
        suffix = None
        match = re.search(r'[0-9]{6}', code)
        
        if not match:
            raise ValueError("Unknown security: {}".format(code))

        number = match.group(0)
        if 'XSHG' in code:
            suffix = 'SH'
        elif 'XSHE' in code:
            suffix = 'SZ'

        if suffix is None:
            suffix = 'SH' if int(number) >= 500000 else 'SZ'
        return number, suffix
def get_security(number,market):
    """
    将文件中分开的交易品种代码和市场重新拼接成聚宽使用的代码形式
    market=1 上海；market=2 深圳
    """
    if market==1:
        security=str(number)+".XSHG"
    elif market==3:
        security=str(number)+".XSHE"
    return security

class Order(object):
    """
    订单类，若回报中无相关域，保持为None
    订单中需要包含账户相关（类型etc.）信息吗
    """
    def __init__(self, status, add_time, is_buy, amount, filled, security,
                 order_id, price, avg_cost, side, action, origin_status, ref, raw,trade_type):
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
    def __init__(self,acct_type=None,acct=None,currency=None,inout_cash=None, available_cash=None, transferable_cash=None, locked_cash=None,
                 margin=None,locked_margin=None, positions=None, long_positions=None, short_positions=None, total_value=None,
                 returns=None, starting_cash=None, positions_value=None, locked_cash_by_purchase=None,
                 locked_cash_by_redeem=None, locked_amount_by_redeem=None, ref=None, raw=None):
        #账户类型
        self.acct_type=acct_type,
        #交易账户
        self.acct=acct,
        #币种 {0,1,2}
        self.currency=currency,
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
        #冻结保证金
        self.locked_margin=locked_margin,
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
    def __init__(self, future_trade_mark=None,prev_position=None,security = None, price = None, acc_avg_cost = None,
                 avg_cost = None, hold_cost = None, init_time = None, transact_time = None,
                 total_amount = None, closeable_amount = None, today_amount = None,
                 locked_amount = None, value = None, side = None, ref=None, raw=None):
        """
        增加对期货品种的标记：
        1.future_trade_mark 投机套保标记（期货交易的不同模式）
        2.prev_position 昨日持仓数
        """
        # 投机套保标记
        self.future_trade_mark=future_trade_mark,
        # 昨仓数
        self.prev_position=prev_position,
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


class TSInterface(ABCInterface):

    Fields = {
        "order": ["orderID", "accountID", "symbol","exchange","Side",
                  "Type", "Price", "Size"],
        "order_report": ["OrderDate", "OrderTime", "FilledDate", "FilledTime",
                         "ClientOrder", "OrderID","Stock","buy/sell","OrderType",
                         "OrderPrice","Quantity","FilledQuantity","LeftQuantity","AvgPrice",
                         "State","Exchange","AccountID"],
        "fund_report": ["Account","Asset"],
        "position_report": ["Account", "Stock", "Quantity", "QuantityAvailable"]
        #                   "MarketCode", "Quantity", "AvlQuantity", "BS", "AvgPrice", "CostValue",
        #                    "PL", "Fare", "AvlFund", "AssertValue", "ValutionPLM2D", "_CostValue",
        #                    "CumulationPL", "CumulationFare"]
    }

    OrderStatusMapping = {
        "0": OrderStatus.CancelSending,
        "1": OrderStatus.CancelReject,
        "2": OrderStatus.OrderNotFound,
        "3": OrderStatus.Canceled,
        "4": OrderStatus.Cancelpending,
        "5": OrderStatus.Expired,
        "6": OrderStatus.Filled,
        "7":OrderStatus.Partiallyfilled,
        "8":OrderStatus.Partiallyfilledurout,
        "9":OrderStatus.Queued,
        "10":OrderStatus.Received,
        "11":OrderStatus.Rejected,
        "12":OrderStatus.Sendfailed,
        "13":OrderStatus.Sending,
        "14":OrderStatus.Sent,
        "15":OrderStatus.unsent
    }

    OrderStatusReverseMapping = {
        OrderStatus.reported: "0",
        OrderStatus.filled: "1",
        OrderStatus.held: "2",
        OrderStatus.ptcanceled:"3",
        OrderStatus.canceled: "4",
        OrderStatus.rejected:"5",
        OrderStatus.unaccepted:"6",
    }
    acct_stock={"0","C","HK0","SHDDF0","SHDDFC","SZDDF0","SZDDFC","S0"}
    acct_future={"A","DHA","DWA","ESA","G2A","GFGA","GJA","GLA","HTA","HUTA","HXA","HZA","JXA","XZA","YAA","ZDA","SA"}
    acct_option={"0B"}

    class LimitOrder(BaseOrder):
        def __init__(self, price):
            self.price = price
            self.type = "L"

    class OptionOPALimitOrder(BaseOrder):
    	#OPA 代表期权限价 FOK
	    def __init__(self, price):
	        self.price = price
	        self.type = "OPA"

    class MarketOrder(BaseOrder):
        def __init__(self):
            self.price = 0
            self.type = "M"

    def __init__(self, order_list_file, order_report_file,
                 fund_report_file, order_held_file):
        self.bucket_mod = False
        self.order_queue = []

        self.csv_files = {
            "order": order_list_file,
            "order_held":order_held_file,
            "order_report": order_report_file,
            "fund_report": fund_report_file,
            "position_report":position_report_file
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

    def _gen_order_from_report(self, ref_df,status=None):
        result = []
        for _, row in ref_df.iterrows():
            rec = row.to_dict()
            #not sure: is_buy,add_time,side
            code,exchange=get_code_exchange(rec["symbol"])
            result.append(Order(status=self._get_order_status(rec["State"]),
                                add_time=pd.to_datetime(rec["OrderTime"], unit="s"),
                                is_buy=rec["buy/sell"],
                                amount=rec["Quantity"],
                                filled=rec["FilledQunatity"],
                                security=".".join(code,exchange),
                                order_id=rec["ClientOrderID"],
                                price=rec["AvgPrice"],
                                avg_cost=rec["AvgPrice"],
                                side="long", action=["close", "open"][str(rec["buy/sell"])=="buy"],
                                origin_status=rec["State"], ref="TS", raw=rec))
        return result

    def _dump(self, template, instruction):
        #文件名格式要求：写入时先使用filename.csv的格式，写完后重命名为filename+I.csv;不同资金账号的订单必须分开;
        #Q:不知道对方系统是追加读取还是直接读取
        instruction.to_csv(self.csv_files[template]+str(instruction["orderID"]), encoding="utf-8", mode="a",
                           header=False, index=False)
        rename(self.csv_files[template]+str(instruction["orderID"]),self.csv_files[template]+str(instruction["orderID"])+"I.csv")
                                            
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
        """
        order()用于下单，将请求写入文件中
        1.type or style?  如果传入不在定义范围内的order类型，则默认为市价单；否则传入定义范围内的order类型
		2.CATS supports shorting futures and options while JoinQuant doesn't
        3.side 重名了,side='long'不变，写文件时变成“Side”
        """
        code, exchange = get_symbol_exchange(security)
        order_style = self.MarketOrder() if not issubclass(type, BaseOrder) else style
        order_id = self.gen_order_id()
        rec = {"orderID":order_id,
               "accountID": kwargs.get("accountID",""),
               "symbol": code,
               "exchange":exchange,
               "Side": ["1", "11"][amount > 0] if not kwargs.get("Side]", False) else kwargs["Side"],
               "type":order_style.type,
               "price": order_style.price,
               "size": abs(int(amount))
             }
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
            #问题：order_list_file是打不开的（重命名了）
            with open(path, "a+") as f:
                #f.seek(0, 2)
                f.seek(0, 0)
                #if self.file_cur_offset[k] < f.tell():
                if True:
                    buffer = StringIO()
                    # f.seek(self.file_cur_offset[k])
                    buffer.write(f.read())
                    buffer.seek(0)
                    self.file_cur_offset[k] = f.tell()
                    df = pd.read_csv(buffer, names=self.Fields[k], memory_map=True, index_col=False)
                    self.df[k] = pd.concat([self.df[k], df])
        # 潜在bug，如果是数值ID会被解析成浮点数转string
        id_mapping = {str(i["ClientOrder"]): str(i["OrderID"]) for i in
                      self.df["order_report"][["ClinetOrder", "OrderID"]].to_dict('records')}
        self.id_mapping.update(id_mapping)

    def get_open_orders(self):
        self.sync()
        df = self.df["order_report"]
        #unaccepted=unreported ??
        df = df[df["State"].astype("str").isin({"CancelSending","Cancelpending","Partiallyfilled","Partiallyfilledurout","Queued","Received","Sending","Sent","unsent"})].groupby(["symbol", "OrderID"]).tail(1)
        return self._gen_order_from_report(df.reset_index())
    
    """
	def get_held_orders(self):
		self.sync()
        df=self.df["order_held"]
        result=[]
        for _, row in df.iterrows():
            rec = row.to_dict()
            #not sure: is_buy,add_time,side
            result.append(Order(status=self._get_order_status(2),
                                add_time=pd.to_datetime(rec["time"], unit="ms"),
                                is_buy=str(rec["tradeside"]) in {"1", "A","F"},
                                amount=rec["fillqty"],
                                filled=rec["fillqty"],
                                security=rec["symbol"],
                                order_id=rec["order_no"],
                                price=rec["fillprice"],
                                avg_cost=rec["fillprice"],
                                action=["close", "open"][str(rec["tradeside"]) in {"1", "A","F"}],
                                ref="CATS", raw=rec,trade_type=rec["trdtype"]))
        return result
    """
    def get_orders(self, order_id=None, security=None, status=None):
        self.sync()
        df = self.df["order_report"]
        if order_id:
            #客户查询订单时使用自己定义的客户委托编号
            df = df[df["ClientOderID"] == order_id]
        if security:
            code,exchange=get_symbol_exchange(security)
            df =df[df["Stock"] == ".".join(code,exchange)]
        if status:
            df = df[df["ord_status"] == status]

        df = df.groupby(["symbol", "client_id"]).tail(1)
        test = self._gen_order_from_report(df.reset_index())
        return test

    def cancel_order(self, order):
        self.sync()
        order_no = self.id_mapping[order.order_id if isinstance(order, Order) else str(order)]
        order_id = self.gen_order_id()
        rec = {"order_no": order_no,
                "accountID":"",
                "symbol":"",
                "exchange":"",
                "side":"-1",
               }
        self._save_order_to_disk(rec)
        return order_id

    @property
    def position(self,acct=None):
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
        df=self.df["position_report"]
        df=df[df["acct"]==str(acct)]
        longpositions=[]
        longname=[]
        for _, row in df.iterrows():
            rec = row.to_dict()
            code,exchange=get_code_exchange(rec["Stock"])
            longname.append(".".join(code,exchange))
            longpositions.append(Position(security=".".join(code,exchange),
                                          total_amount=rec["Quantity"],
                                          closeable_amount=rec["QuantityAvailable"],
                                          ref="TS",raw=rec))
        return dict(zip(longname,longpositions))
    
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
        result=[]
        df=self.df["fund_report"]
        for _, row in df.iterrows():
            rec = row.to_dict()
            positions=self.position(acct=rec["Account"])
            longpositions=positions
            result.append(Portfolio(acct=rec["Account"],total_value=rec["Asset"],
                                    margin=rec["Asset"],positions=positions,
                                    long_positions=longpositions,
            						ref="TS",raw=rec))

        return result
    #融资买入
    def margincash_open(self, security, amount, style=None, **kwargs):
        return self.order(security=security, amount=amount, style=style,
                          Side="11", **kwargs)
    #卖券还款
    def margincash_close(self, security, amount, style=None, **kwargs):
        return self.order(security=security, amount=amount, style=style,
                          Side="12", **kwargs)
    #融券卖出
    def marginsec_open(self, security, amount, style=None, **kwargs):
        return self.order(security=security, amount=amount, style=style,
                          Side="14", **kwargs)
    #买券还券
    def marginsec_close(self, security, amount, style=None, **kwargs):
        return self.order(security=security, amount=amount, style=style,
                          Side="13", **kwargs)
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
    
    today=dt.date.today().strftime('%y%m%d')
    t =TSInterface(order_list_file="./OrderList",
                   order_report_file="./Order"+str(today)+".csv",
                   fund_report_file="./Account.csv",
                   position_report_file="./Position.csv")
    t.get_orders()

