import sys, typing, csv, argparse
import datetime
import decimal
import sqlite3
import readline
import re
from dataclasses import dataclass
from enum import Enum

D = decimal.Decimal

DATE_FMT = '%Y-%m-%d %H:%M:%S'
cC = '\x1b[0m'
cLog = '\x1b[38;5;8m'
cInterest = '\x1b[38;5;2m'
cImportant = '\x1b[38;5;1m'

# @dataclass
# class Chain:

@dataclass
class Processed:
    trade_id: int
    trail_id: int
    comment: str

@dataclass
class PositionSummary:
    id: int
    category: str
    currency: str
    symbol: str
    quantity: decimal.Decimal
    mult: decimal.Decimal
    cost_price: decimal.Decimal
    cost_basis: decimal.Decimal
    close_price: decimal.Decimal
    value: decimal.Decimal
    unrealized_pl: decimal.Decimal
    code: str

    def __init__(self,id,cat,curr,symnol,quantity,mult,price,basis,close,value,unrlpl,code):
        self.id=id
        self.category=cat
        self.currency=curr
        self.symbol=symnol
        self.quantity=quantity
        self.mult=mult
        self.cost_price=price
        self.cost_basis=basis
        self.close_price=close
        self.value=value
        self.unrealized_pl=unrlpl
        self.code=code

    def __str__(self) -> str:
        return f"{self.currency} {self.symbol:20} {self.quantity:8} x {self.mult:3} cost_price {round(self.cost_price,2):8} "\
               f"cost_basis {round(self.cost_basis,2):6} close_price {round(self.close_price,2):9} "\
               f"value {round(self.value,2):9} unrl p/l {round(self.unrealized_pl,2):8} {self.code}"

@dataclass
class Trade:
    id: int
    symbol: str
    date: datetime
    quantity: decimal.Decimal
    price: decimal.Decimal
    proceeds: decimal.Decimal
    comm: decimal.Decimal
    basis: decimal.Decimal
    pl: decimal.Decimal
    code: str

    def __init__(self,id_,sym,date_str,q,pri,pro,co,bas,pl_,cod):
        self.id=id_
        self.symbol=sym
        self.date = datetime.datetime.strptime(date_str, DATE_FMT) if isinstance(date_str,str) else date_str
        self.quantity=int(q)
        self.price=D(pri)
        self.proceeds=D(pro)
        self.comm=D(co)
        self.basis=D(bas)
        self.pl=D(pl_)
        self.code=cod

    def is_open(self): return self.code.find('O') != -1
    def is_close(self): return self.code.find('C') != -1
    def is_expired(self): return self.code.find('Ep') != -1
    def is_put(self): return self.symbol[-1] == 'P'
    def is_call(self): return self.symbol[-1] == 'C'
    def get_strike(self): return self.symbol.split(' ')[2]
    def get_opt(self): return self.symbol[-1]
    def get_strikeopt(self): return self.symbol.split(' ', 1)[1]

    def __str__(self) -> str:
        return f"{self.symbol:20} {self.date} quantity {self.quantity:3} price {self.price:8} proceeds {round(self.proceeds,2):8} comm/fee {round(self.comm,2):6} basis {round(self.basis,2):9} p/l {round(self.pl,2):8} {self.code}"
    
    def get_type(self) -> str:
        if self.is_open(): return "OPEN"
        elif self.is_close(): return "EXPIRE" if self.is_expired() else "CLOSE"

TrailType = Enum('TRAIL', ['Unknown', 'Open', 'Close', 'Expire', 'Roll', 'ShortStrangle', 'Butterfly'])

@dataclass
class TradeInfo:
    active: bool = True # 

@dataclass
class Trail:

    type: TrailType
    trades: list[tuple[Trade, TradeInfo]]
    balance: int # amount of assets sold and bought
    # open: bool

    def __init__(self):
        # self.open = True
        self.trades = []
        self.type = TrailType.Unknown
        self.balance = 0

    def calc_balance(self) -> dict[str, decimal.Decimal]:
        bal = {}
        for t, i in self.trades:
            bal[t.symbol] = bal.get(t.symbol, 0) + t.quantity
        return bal
    
    def has_of_date(self, d) -> bool:
        for t, i in self.trades:
            if t.date == d: return True
        return False

def parse_lines(f, field_delim: str = ',') -> list[list[str]]:
    reader = csv.reader(f, delimiter=field_delim)
    lines = []
    for row in reader:
        lines.append(row)
    return lines

def parse_datetimestr(dt: str) -> datetime.datetime:
    d_s, t_s = dt.split(',')
    d = datetime.date.fromisoformat(d_s.strip())
    t = datetime.time.fromisoformat(t_s.strip())
    return datetime.datetime.combine(d, t)


# filter: a dict 
def parse_tables(conn, lines, field_delim: str = ','):

        mode = 0 # 0: search-header 1: parse-table
        pos: dict

        cur : sqlite3.Cursor = conn.cursor()

        option_trades = { 'DataDiscriminator': 'Order', 'Asset Category': 'Equity and Index Options' }
        open_positions = { 'DataDiscriminator': 'Summary', 'Header': 'Data' }

        cur.execute('CREATE TABLE processed('\
            'trade_id INTEGER, '\
            'trail_id INTEGER, '\
            'comment TEXT,'\
            'FOREIGN KEY (trade_id)'\
            'REFERENCES trades (id)'\
                'ON DELETE SET NULL'\
            ');')

        cur.execute('CREATE TABLE trades('\
            'id INTEGER PRIMARY KEY,'\
            'symbol TEXT,'\
            'date DATETIME,'\
            'quantity DECIMAL(10, 4),'\
            'price TEXT,'\
            'proceeds TEXT,'\
            'comm TEXT,'\
            'basis TEXT,'\
            'pl TEXT,'\
            'code TEXT'\
            ');')

        cur.execute('CREATE TABLE positions('\
            'id INTEGER PRIMARY KEY,'\
            'category TEXT,'\
            'currency TEXT,'\
            'symbol TEXT,'\
            'quantity DECIMAL(10, 4),'\
            'mult DECIMAL(10, 4),'\
            'cost_price TEXT,'\
            'cost_basis TEXT,'\
            'close_price TEXT,'\
            'value TEXT,'\
            'unrealized_pl TEXT,'\
            'code TEXT'\
            ');')

        tables = {
            'Trades':           { 'id': 1, 'filter': option_trades },
            'Open Positions':   { 'id': 2, 'filter': open_positions }
        }

        proc_tab = ""
        proc_filter = None

        for line in lines:
            if mode == 0:
                proc_tab = line[0]
                if proc_tab in tables and line[1] == 'Header':
                    itm_idx = {}
                    for i, item in enumerate(line): # create a mapping of column name -> column index
                        itm_idx[item] = i
                    mode = tables[ proc_tab ][ 'id' ]
                    proc_filter = tables[ proc_tab ][ 'filter' ]
                    continue

            if mode != 0:
                # are we finished with the current table?
                if line[0] != proc_tab:
                    mode = 0
                    proc_tab = ""
                    proc_filter = None
                    continue

                # filter out rows we're not interested in
                include = True
                if filter != None:
                    for k, v in proc_filter.items():
                        if k in itm_idx:
                            col = itm_idx[k]
                            if line[col] != v:
                                include = False
                                break

                if not include:
                    continue

                if mode == 1:
                    dt = parse_datetimestr(line[ itm_idx['Date/Time'] ])
                    cur.execute(f"INSERT INTO trades(symbol,date,quantity,price,proceeds,comm,basis,pl,code)\n"\
                        "VALUES("\
                                f"'{  line[ itm_idx['Symbol'] ]}',"\
                                f"'{  dt}',"\
                                f"'{  line[ itm_idx['Quantity'] ]}',"\
                                f"'{D(line[ itm_idx['T. Price'] ]).to_eng_string()}',"\
                                f"'{D(line[ itm_idx['Proceeds'] ]).to_eng_string()}',"\
                                f"'{D(line[ itm_idx['Comm/Fee'] ]).to_eng_string()}',"\
                                f"'{D(line[ itm_idx['Basis'] ]).to_eng_string()}',"\
                                f"'{D(line[ itm_idx['Realized P/L'] ]).to_eng_string()}',"\
                                f"'{  line[ itm_idx['Code'] ]}'"\
                                ");"
                            )
                elif mode == 2:
                    cur.execute(f"INSERT INTO positions(category,currency,symbol,quantity,mult,cost_price,"\
                                    "cost_basis,close_price,value,unrealized_pl,code)\n"\
                        "VALUES("\
                                f"'{  line[ itm_idx['Asset Category'] ]}',"\
                                f"'{  line[ itm_idx['Currency'] ]}',"\
                                f"'{  line[ itm_idx['Symbol'] ]}',"\
                                f"'{  line[ itm_idx['Quantity'] ]}',"\
                                f"'{D(line[ itm_idx['Mult'] ]).to_eng_string()}',"\
                                f"'{D(line[ itm_idx['Cost Price'] ]).to_eng_string()}',"\
                                f"'{D(line[ itm_idx['Cost Basis'] ]).to_eng_string()}',"\
                                f"'{D(line[ itm_idx['Close Price'] ]).to_eng_string()}',"\
                                f"'{D(line[ itm_idx['Value'] ]).to_eng_string()}',"\
                                f"'{D(line[ itm_idx['Unrealized P/L'] ]).to_eng_string()}',"\
                                f"'{  line[ itm_idx['Code'] ]}'"\
                                ");"
                            )


def dump_db(conn: sqlite3.Connection):
    # for t in self._trades:
    #     print(t)

    pass

def dump_tables(conn: sqlite3.Connection, tabs = None):
    cur: sqlite3.Cursor = conn.cursor()

    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cur.fetchall()
        # print(f"TABLES: {tables}")
        for t in tables:
            if tabs != None and not t[0] in tabs:
                continue
            # print(f"TABLE: {t[0]}")
            cur.execute(f"SELECT * FROM {t[0]}")
            dat = cur.fetchall()
            for d in dat:
                print(f"{d}")
    except sqlite3.Error as e:
        print(e)


def db_conn(conn : sqlite3.Connection):

    print("/tables show   tables in db\n"\
          "/info <table>  info on table\n")
    
    try:

        cur = conn.cursor()
        while True:
            line = input("> ")

            # hijack special commands
            if line.startswith('/'):
                if line.find('/tables') != -1:
                    line = "SELECT name FROM sqlite_master WHERE type='table';"
                elif line.find('/info') != -1:
                    line = f"PRAGMA table_info({line.split()[1]})"
                    
            try:
                cur.execute(line)
                dat = cur.fetchall()
                for d in dat:
                    print(f"{d}")
            except sqlite3.Error as e:
                print(e)

    except (EOFError, KeyboardInterrupt) as e:
        print("done")

def find_trail2(trails: list[Trail], trade: Trade) -> int:
    for trail_idx, trail in enumerate(trails):

        # if not trail.open:
        #     continue

        bal = trail.calc_balance()

        if trail.has_of_date(trade.date):
            if not trade.is_expired():
                return trail_idx

        # closing trade
        if trade.is_close():
            # if trail contains this symbol and it's not "closed" yet, add to it
            if trade.symbol in bal and bal[trade.symbol] != decimal.Decimal(0):
                return trail_idx

        # opening more of the same and this trail isn't closed on this symbol
        elif trade.symbol in bal and bal[trade.symbol] != 0:
            return trail_idx

        # check for a "options construct"
        for ttrade, tinfo in trail.trades:

            if trade.date == ttrade.date:
                if trade.symbol in bal:
                # and so if it's not 'expiration', then this trade is related to this trail
                    return trail_idx

    return -1

def sort_trades(conn: sqlite3.Connection) -> list[Trail]:

    cur : sqlite3.Cursor = conn.cursor()
    cur.execute("SELECT * FROM trades ORDER BY date")
    trades = [ Trade(*t) for t in cur.fetchall() ]
    trails: list[Trail] = []

    for idx, trade in enumerate(trades):

        print(f"{cLog}#{idx} TRADE id {trade.id:03} {trade.quantity} {trade.symbol} {trade.get_type()} {trade.date}{cC}")

        if trade.symbol == "SPX 19MAY23 3570 P":
            print("ATTENTION!")

        # find Trail
        trail:Trail = None
        trail_idx = find_trail2(trails, trade)
        if trail_idx == -1:
            trail_idx = len(trails)
            print(f"{cLog}  creating trail {trail_idx}{cC}")
            trail = Trail()
            trails.append(trail)
        else:
            trail = trails[trail_idx]
            print(f"{cLog}  using trail {trail_idx}{cC}")

        trail.trades.append((trade, TradeInfo))
        trail.balance += trade.quantity

    return trails


def detect_type(l: list[Trade]):
    if len(l) == 0:
        return TrailType.Unknown

    # test if all expired or closed
    if sum(map(lambda x: x.is_expired(), l)) == len(l):
        return TrailType.Expire
    if sum(map(lambda x: x.is_close(), l)) == len(l):
        return TrailType.Close

    if len(l) == 1:
        if l[0].is_open(): return TrailType.Open
        elif l[0].is_close(): return TrailType.Close
        return TrailType.Unknown
    
    if len(l) >= 2:
        if (l[0].is_open() and l[1].is_close()) or (l[0].is_close() and l[1].is_open()):
            return TrailType.Roll
        elif l[0].is_open() and l[1].is_open() and ((l[0].is_put() and l[1].is_call()) or (l[0].is_call() and l[1].is_put())):
            assert(l[0].quantity < 0 and l[1].quantity < 0)
            assert(l[0].quantity == l[1].quantity)
            return TrailType.ShortStrangle
        else:
            return TrailType.Unknown


def exam_trail(trail: Trail):

    proceeds: decimal.Decimal = decimal.Decimal(0)
    comm: decimal.Decimal = decimal.Decimal(0)
    pl: decimal.Decimal = decimal.Decimal(0)
    i: int = 0
    while ( i <len(trail.trades) ):
        cur: Trade = trail.trades[i][0]
        # next: Trade = trail.trades[i+1][0] if i < len(trail.trades) else None
        # if cur.date == next.date and not cur.is_expired():
        of_same_date = [ x[0] for x in trail.trades[i:] if x[0].date == cur.date ]
        type = detect_type(of_same_date)
        print(f"  {type.name}:", end='')
        if type == TrailType.Roll:
            close_first = sorted(of_same_date, key=lambda x: x.is_open())
            print(f" {close_first[0].get_strikeopt()} -> {close_first[1].get_strikeopt()}", end='')

        if type == TrailType.Roll:
            print(f" {(close_first[0].proceeds+close_first[0].comm) + close_first[1].proceeds+close_first[1].comm}", end='')

        print(f" {sum(map(lambda x: x.pl, of_same_date))}")

        for x in of_same_date:
            pl += x.pl
            proceeds += x.proceeds
            comm += x.comm
        i = i + len(of_same_date)

        # else:
        #     closing = [ x for x in trail.trades[i:] if x.is_close ]

    print(f"  {cInterest}Total P/L: {pl}{cC}")
    print(f"  {cInterest}Total Proceeds: {proceeds} Comm/Fee: {comm} -> {proceeds + comm}{cC}")

if __name__ == '__main__':

    ap = argparse.ArgumentParser(
        prog='IBKR analyzer',
        description='Analyzes trades in various aspects')
    ap.add_argument('filename', action='store', nargs='?', default="hist.csv", help="history filename")
    ap.add_argument('-i', '--interactive', action='store_true', help="enter interactive mode")
    ap.add_argument('-m', '--mode', choices=['roll','pl'], action='store', help="analyze mode")
    ap.add_argument('-d', '--date', action='store', help="date or date range in iso formats eg 2022-01-01 : 2023-03-29")
    ap.add_argument('-t', '--ticker', action='store', help="ticker regex")

    dt_from: datetime.datetime = None #datetime.datetime.now().replace(month=1, day=1)
    dt_to: datetime.datetime = None #datetime.datetime.now()

    t_re = None

    args = ap.parse_args()

    if args.ticker:
        t_re = re.compile(args.ticker)

    in_f = "hist.csv"
    if args.filename:
        in_f = args.filename
        print(f"reading {in_f}")

    lines = []
    with open(in_f, "r") as f:
        lines = parse_lines(f)

    if args.date != None:
        dt_from = datetime.datetime.fromisoformat(args.date)

    conn : sqlite3.Connection = sqlite3.connect("file::memory:?cache=shared", uri=True)
    parse_tables(conn, lines)
    dump_db(conn)

    if args.interactive:
        print("Enter interactive mode\n")
        db_conn(conn)
    else:
        pass

    if args.mode == 'roll':
        trails = sort_trades(conn)

        cur : sqlite3.Cursor = conn.cursor()
        for trail_idx, trail in enumerate(trails):

            if t_re:
                if not re.match(t_re, trail.trades[0][0].symbol):
                    continue

            if trail.balance != 0: print(f"{cImportant}", end='')
            print(f"#{trail_idx}: balance {trail.balance}{cC}")


            for trade, _ in trail.trades:
                cur.execute(f"SELECT * FROM trades WHERE id = {trade.id} LIMIT 1")
                trade = Trade(*cur.fetchone())
                print(f"  {cLog}{trade}{cC}")

            exam_trail(trail)

            print("")

    elif args.mode == 'pl':

        cur : sqlite3.Cursor = conn.cursor()
        cur.execute("SELECT * FROM trades ORDER BY date")
        trades = [ Trade(*t) for t in cur.fetchall() ]

        pl: D = D(0)
        proceeds: D = D(0)
        for trade in trades:
            if dt_from and trade.date < dt_from: continue
            if dt_to and trade.date > dt_to: continue
            if t_re:
                if not re.match(t_re, trade.symbol):
                    continue

            pl += trade.pl
            proceeds += trade.proceeds

        print(f"{cInterest}Total Proceeds: {proceeds}{cC}")
        print(f"{cInterest}Total P/L: {pl}{cC}")

