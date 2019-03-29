参照券商PB系统的说明文件，实现了部分系统的“文件单”对接程序。



**注：说明文件也许与真实系统不一致。相关文件并没有使用真实软件测试，并不保证文件的完全可用性。如遇问题请自行修改代码。**



**文件**

- interface_CATS：中信CATS
- interface_DTS：广发DTS
- interface_TS：国信TS
- interface_XT：迅投PB
- interface_ZS：招商交易大师



**示例**

```python
from interface_DTS import *

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

```



