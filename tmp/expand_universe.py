"""
Builds sector_universe.csv with exactly 250 high-liquidity equities (all Nifty 500 / F&O-eligible)
+ 12 sector indices = 262 rows total.
Run: .venv\Scripts\python.exe tmp\expand_universe.py
"""
import pandas as pd
from pathlib import Path

OUTPUT = Path("config_data/sector_universe.csv")

# Criteria: Nifty 500 member OR F&O-eligible OR avg daily turnover > ₹50 Cr
EQUITIES = [
    # ─── ENERGY (20) ──────────────────────────────────────────────────────────
    ("RELIANCE",   "NSE_EQ|INE002A01018", "Energy"),
    ("ONGC",       "NSE_EQ|INE213A01029", "Energy"),
    ("BPCL",       "NSE_EQ|INE029A01011", "Energy"),
    ("IOC",        "NSE_EQ|INE242A01010", "Energy"),
    ("GAIL",       "NSE_EQ|INE129A01019", "Energy"),
    ("HINDPETRO",  "NSE_EQ|INE094A01015", "Energy"),
    ("ADANIENT",   "NSE_EQ|INE423A01024", "Energy"),
    ("ADANIGREEN", "NSE_EQ|INE364U01010", "Energy"),
    ("ADANIPORTS", "NSE_EQ|INE742F01042", "Energy"),
    ("TATAPOWER",  "NSE_EQ|INE245A01021", "Energy"),
    ("NTPC",       "NSE_EQ|INE733E01010", "Energy"),
    ("POWERGRID",  "NSE_EQ|INE752E01010", "Energy"),
    ("TORNTPOWER", "NSE_EQ|INE813H01021", "Energy"),
    ("CESC",       "NSE_EQ|INE486A01013", "Energy"),
    ("PETRONET",   "NSE_EQ|INE347G01014", "Energy"),
    ("GSPL",       "NSE_EQ|INE246F01010", "Energy"),
    ("JSWENERGY",  "NSE_EQ|INE121E01018", "Energy"),
    ("NHPC",       "NSE_EQ|INE848E01016", "Energy"),
    ("SJVN",       "NSE_EQ|INE002L01015", "Energy"),
    ("ADANITRANS", "NSE_EQ|INE931S01010", "Energy"),

    # ─── BANKING (24) ─────────────────────────────────────────────────────────
    ("HDFCBANK",   "NSE_EQ|INE040A01034", "Banking"),
    ("ICICIBANK",  "NSE_EQ|INE090A01021", "Banking"),
    ("SBIN",       "NSE_EQ|INE062A01020", "Banking"),
    ("KOTAKBANK",  "NSE_EQ|INE237A01036", "Banking"),
    ("AXISBANK",   "NSE_EQ|INE238A01034", "Banking"),
    ("INDUSINDBK", "NSE_EQ|INE095A01012", "Banking"),
    ("BANKBARODA", "NSE_EQ|INE028A01039", "Banking"),
    ("PNB",        "NSE_EQ|INE160A01022", "Banking"),
    ("FEDERALBNK", "NSE_EQ|INE171A01029", "Banking"),
    ("IDFCFIRSTB", "NSE_EQ|INE092T01019", "Banking"),
    ("AUBANK",     "NSE_EQ|INE949L01017", "Banking"),
    ("BANDHANBNK", "NSE_EQ|INE545U01014", "Banking"),
    ("CANBK",      "NSE_EQ|INE476A01014", "Banking"),
    ("UNIONBANK",  "NSE_EQ|INE692A01016", "Banking"),
    ("INDIANB",    "NSE_EQ|INE562A01011", "Banking"),
    ("CENTRALBK",  "NSE_EQ|INE483A01010", "Banking"),
    ("RBLBANK",    "NSE_EQ|INE976G01028", "Banking"),
    ("YESBANK",    "NSE_EQ|INE528G01035", "Banking"),
    ("DCBBANK",    "NSE_EQ|INE503A01015", "Banking"),
    ("CSBBANK",    "NSE_EQ|INE679A01013", "Banking"),
    ("KARURVYSYA", "NSE_EQ|INE036D01028", "Banking"),
    ("MAHABANK",   "NSE_EQ|INE457A01014", "Banking"),   # F&O eligible PSU
    ("JKBANK",     "NSE_EQ|INE168A01041", "Banking"),   # F&O eligible
    ("SOUTHBANK",  "NSE_EQ|INE683A01023", "Banking"),   # Nifty 500

    # ─── IT (18) ──────────────────────────────────────────────────────────────
    ("TCS",        "NSE_EQ|INE467B01029", "IT"),
    ("INFY",       "NSE_EQ|INE009A01021", "IT"),
    ("WIPRO",      "NSE_EQ|INE075A01022", "IT"),
    ("HCLTECH",    "NSE_EQ|INE860A01027", "IT"),
    ("TECHM",      "NSE_EQ|INE669C01036", "IT"),
    ("LTIM",       "NSE_EQ|INE214T01019", "IT"),
    ("MPHASIS",    "NSE_EQ|INE356A01018", "IT"),
    ("COFORGE",    "NSE_EQ|INE591G01025", "IT"),
    ("PERSISTENT", "NSE_EQ|INE262H01021", "IT"),
    ("LTTS",       "NSE_EQ|INE010V01017", "IT"),
    ("KPITTECH",   "NSE_EQ|INE793A01012", "IT"),
    ("OFSS",       "NSE_EQ|INE881D01027", "IT"),
    ("MASTEK",     "NSE_EQ|INE234B01023", "IT"),
    ("TATAELXSI",  "NSE_EQ|INE670A01012", "IT"),
    ("BIRLASOFT",  "NSE_EQ|INE836B01017", "IT"),       # Nifty 500 / F&O
    ("CYIENT",     "NSE_EQ|INE136B01020", "IT"),       # Nifty 500 / F&O
    ("TANLA",      "NSE_EQ|INE483B01026", "IT"),       # Nifty 500
    ("ROUTE",      "NSE_EQ|INE450U01017", "IT"),       # F&O eligible comms

    # ─── TELECOM (3) ──────────────────────────────────────────────────────────
    ("BHARTIARTL", "NSE_EQ|INE397D01024", "Telecom"),
    ("IDEA",       "NSE_EQ|INE669E01016", "Telecom"),
    ("TATACOMM",   "NSE_EQ|INE151A01013", "Telecom"),

    # ─── FMCG (18) ────────────────────────────────────────────────────────────
    ("HINDUNILVR", "NSE_EQ|INE030A01027", "FMCG"),
    ("ITC",        "NSE_EQ|INE154A01025", "FMCG"),
    ("NESTLEIND",  "NSE_EQ|INE239A01024", "FMCG"),
    ("BRITANNIA",  "NSE_EQ|INE216A01030", "FMCG"),
    ("DABUR",      "NSE_EQ|INE016A01026", "FMCG"),
    ("MARICO",     "NSE_EQ|INE196A01026", "FMCG"),
    ("COLPAL",     "NSE_EQ|INE259A01022", "FMCG"),
    ("GODREJCP",   "NSE_EQ|INE102D01028", "FMCG"),
    ("TATACONSUM", "NSE_EQ|INE192A01025", "FMCG"),
    ("UBL",        "NSE_EQ|INE686F01025", "FMCG"),
    ("MCDOWELL",   "NSE_EQ|INE854D01024", "FMCG"),
    ("VBL",        "NSE_EQ|INE200M01039", "FMCG"),
    ("EMAMILTD",   "NSE_EQ|INE548C01032", "FMCG"),
    ("BAJAJCON",   "NSE_EQ|INE933K01021", "FMCG"),
    ("JYOTHYLAB",  "NSE_EQ|INE668F01031", "FMCG"),
    ("RADICO",     "NSE_EQ|INE944F01028", "FMCG"),     # Nifty 500 / F&O
    ("ZYDUSWELL",  "NSE_EQ|INE729I01019", "FMCG"),     # Nifty 500
    ("PGHH",       "NSE_EQ|INE179A01014", "FMCG"),     # P&G India

    # ─── PHARMA (22) ──────────────────────────────────────────────────────────
    ("SUNPHARMA",  "NSE_EQ|INE044A01036", "Pharma"),
    ("DRREDDY",    "NSE_EQ|INE089A01031", "Pharma"),
    ("CIPLA",      "NSE_EQ|INE059A01026", "Pharma"),
    ("DIVISLAB",   "NSE_EQ|INE361B01024", "Pharma"),
    ("AUROPHARMA", "NSE_EQ|INE406A01037", "Pharma"),
    ("BIOCON",     "NSE_EQ|INE376G01013", "Pharma"),
    ("LUPIN",      "NSE_EQ|INE326A01037", "Pharma"),
    ("TORNTPHARM", "NSE_EQ|INE685A01028", "Pharma"),
    ("APOLLOHOSP", "NSE_EQ|INE437A01024", "Pharma"),
    ("LALPATHLAB", "NSE_EQ|INE600L01024", "Pharma"),
    ("ZYDUSLIFE",  "NSE_EQ|INE010B01027", "Pharma"),
    ("GLENMARK",   "NSE_EQ|INE935A01035", "Pharma"),
    ("IPCA",       "NSE_EQ|INE571A01020", "Pharma"),
    ("NATCOPHARM", "NSE_EQ|INE987B01026", "Pharma"),
    ("ALKEM",      "NSE_EQ|INE540L01014", "Pharma"),
    ("GRANULES",   "NSE_EQ|INE101D01020", "Pharma"),
    ("LAURUSLABS", "NSE_EQ|INE688L01026", "Pharma"),
    ("METROPOLIS", "NSE_EQ|INE131I01038", "Pharma"),
    ("AJANTPHARM", "NSE_EQ|INE031B01049", "Pharma"),   # Nifty 500 / F&O
    ("JBCHEPHARM", "NSE_EQ|INE445A01011", "Pharma"),   # Nifty 500
    ("MANKIND",    "NSE_EQ|INE634H01001", "Pharma"),   # Nifty 500 high volume
    ("ERIS",       "NSE_EQ|INE406A01045", "Pharma"),   # Nifty 500

    # ─── AUTO (21) ────────────────────────────────────────────────────────────
    ("TATAMOTORS", "NSE_EQ|INE155A01022", "Auto"),
    ("M&M",        "NSE_EQ|INE101A01026", "Auto"),
    ("MARUTI",     "NSE_EQ|INE585B01010", "Auto"),
    ("BAJAJ-AUTO", "NSE_EQ|INE917I01010", "Auto"),
    ("EICHERMOT",  "NSE_EQ|INE066A01021", "Auto"),
    ("HEROMOTOCO", "NSE_EQ|INE158A01026", "Auto"),
    ("TVSMOTOR",   "NSE_EQ|INE494B01023", "Auto"),
    ("ASHOKLEY",   "NSE_EQ|INE208A01029", "Auto"),
    ("BALKRISIND", "NSE_EQ|INE787D01026", "Auto"),
    ("MRF",        "NSE_EQ|INE883A01011", "Auto"),
    ("MOTHERSON",  "NSE_EQ|INE775A01035", "Auto"),
    ("BOSCHLTD",   "NSE_EQ|INE323A01026", "Auto"),
    ("CEATLTD",    "NSE_EQ|INE482A01020", "Auto"),
    ("APOLLOTYRE", "NSE_EQ|INE438A01022", "Auto"),
    ("ESCORTS",    "NSE_EQ|INE042A01014", "Auto"),
    ("BHARATFORG", "NSE_EQ|INE465A01025", "Auto"),
    ("EXIDEIND",   "NSE_EQ|INE302A01020", "Auto"),
    ("AMARAJABAT", "NSE_EQ|INE885A01032", "Auto"),
    ("SUNDRMFAST", "NSE_EQ|INE057B01024", "Auto"),     # Nifty 500 / F&O
    ("WABCOINDIA", "NSE_EQ|INE342A01014", "Auto"),     # Nifty 500
    ("MAHINDCIE",  "NSE_EQ|INE536H01010", "Auto"),     # CIE Automotive Nifty500

    # ─── METALS (15) ──────────────────────────────────────────────────────────
    ("TATASTEEL",  "NSE_EQ|INE081A01020", "Metals"),
    ("JSWSTEEL",   "NSE_EQ|INE019A01038", "Metals"),
    ("HINDALCO",   "NSE_EQ|INE038A01020", "Metals"),
    ("COALINDIA",  "NSE_EQ|INE522F01014", "Metals"),
    ("VEDL",       "NSE_EQ|INE205A01025", "Metals"),
    ("NMDC",       "NSE_EQ|INE584A01023", "Metals"),
    ("NATIONALUM", "NSE_EQ|INE139A01034", "Metals"),
    ("SAIL",       "NSE_EQ|INE114A01011", "Metals"),
    ("JINDALSTEL", "NSE_EQ|INE220G01021", "Metals"),
    ("APLAPOLLO",  "NSE_EQ|INE702C01027", "Metals"),
    ("RATNAMANI",  "NSE_EQ|INE703B01027", "Metals"),
    ("WELCORP",    "NSE_EQ|INE191B01025", "Metals"),
    ("MOIL",       "NSE_EQ|INE490G01020", "Metals"),
    ("HINDCOPPER", "NSE_EQ|INE531E01026", "Metals"),
    ("TATACHEM",   "NSE_EQ|INE092A01019", "Metals"),   # Nifty 500 / F&O chemicals

    # ─── CEMENT (10) ──────────────────────────────────────────────────────────
    ("ULTRACEMCO", "NSE_EQ|INE481G01011", "Cement"),
    ("SHREECEM",   "NSE_EQ|INE070A01015", "Cement"),
    ("AMBUJACEM",  "NSE_EQ|INE079A01024", "Cement"),
    ("ACC",        "NSE_EQ|INE012A01025", "Cement"),
    ("DALBHARAT",  "NSE_EQ|INE531A01024", "Cement"),
    ("RAMCOCEM",   "NSE_EQ|INE331A01037", "Cement"),
    ("JKCEMENT",   "NSE_EQ|INE823G01014", "Cement"),
    ("HEIDELBERG", "NSE_EQ|INE578A01019", "Cement"),
    ("BIRLACORPN", "NSE_EQ|INE340A01012", "Cement"),   # Nifty 500
    ("INDIACEM",   "NSE_EQ|INE383A01012", "Cement"),   # Nifty 500 / F&O

    # ─── INFRASTRUCTURE (25) ──────────────────────────────────────────────────
    ("LT",         "NSE_EQ|INE018A01030", "Infrastructure"),
    ("SIEMENS",    "NSE_EQ|INE003A01024", "Infrastructure"),
    ("ABB",        "NSE_EQ|INE117A01022", "Infrastructure"),
    ("HAVELLS",    "NSE_EQ|INE176B01034", "Infrastructure"),
    ("BHEL",       "NSE_EQ|INE257A01026", "Infrastructure"),
    ("BEL",        "NSE_EQ|INE263A01024", "Infrastructure"),
    ("HAL",        "NSE_EQ|INE066F01020", "Infrastructure"),
    ("IRCTC",      "NSE_EQ|INE335Y01020", "Infrastructure"),
    ("PIIND",      "NSE_EQ|INE603J01030", "Infrastructure"),
    ("DIXON",      "NSE_EQ|INE935N01020", "Infrastructure"),
    ("CUMMINSIND", "NSE_EQ|INE298A01020", "Infrastructure"),
    ("KEC",        "NSE_EQ|INE389H01022", "Infrastructure"),
    ("KALPATPOWR", "NSE_EQ|INE220J01025", "Infrastructure"),
    ("THERMAX",    "NSE_EQ|INE152A01029", "Infrastructure"),
    ("GRINDWELL",  "NSE_EQ|INE536A01023", "Infrastructure"),
    ("AIAENG",     "NSE_EQ|INE212H01026", "Infrastructure"),
    ("TIINDIA",    "NSE_EQ|INE289B01019", "Infrastructure"),
    ("CONCOR",     "NSE_EQ|INE111A01025", "Infrastructure"),
    ("IRFC",       "NSE_EQ|INE053F01010", "Infrastructure"),
    ("RVNL",       "NSE_EQ|INE415G01027", "Infrastructure"),
    ("POLYCAB",    "NSE_EQ|INE455K01017", "Infrastructure"),  # Nifty 500 / F&O
    ("SUZLON",     "NSE_EQ|INE040H01021", "Infrastructure"),  # Nifty 500 / F&O
    ("JSWINFRA",   "NSE_EQ|INE900P01016", "Infrastructure"),  # Nifty 500
    ("MAZDOCK",    "NSE_EQ|INE249Z01012", "Infrastructure"),  # Defense high volume
    ("ASTRAL",     "NSE_EQ|INE006I01046", "Infrastructure"),  # Nifty 500 / F&O pipes

    # ─── FINANCE (28) ──────────────────────────────────────────────────────────
    ("SBILIFE",    "NSE_EQ|INE123W01016", "Finance"),
    ("HDFCLIFE",   "NSE_EQ|INE795G01014", "Finance"),
    ("ICICIPRULI", "NSE_EQ|INE726G01019", "Finance"),
    ("BAJFINANCE", "NSE_EQ|INE296A01032", "Finance"),
    ("BAJAJFINSV", "NSE_EQ|INE918I01026", "Finance"),
    ("CHOLAFIN",   "NSE_EQ|INE121A01024", "Finance"),
    ("MUTHOOTFIN", "NSE_EQ|INE414G01012", "Finance"),
    ("M&MFIN",     "NSE_EQ|INE774D01024", "Finance"),
    ("MANAPPURAM", "NSE_EQ|INE522D01027", "Finance"),
    ("SHRIRAMFIN", "NSE_EQ|INE721A01047", "Finance"),
    ("SBICARD",    "NSE_EQ|INE018E01016", "Finance"),
    ("RECLTD",     "NSE_EQ|INE020B01018", "Finance"),
    ("PFC",        "NSE_EQ|INE134E01011", "Finance"),
    ("CANFINHOME", "NSE_EQ|INE477A01020", "Finance"),
    ("ICICIGI",    "NSE_EQ|INE765G01017", "Finance"),
    ("NIACL",      "NSE_EQ|INE470Y01017", "Finance"),
    ("ABCAPITAL",  "NSE_EQ|INE674K01013", "Finance"),
    ("LICIHSGFIN", "NSE_EQ|INE115A01026", "Finance"),
    ("POONAWALLA", "NSE_EQ|INE511C01022", "Finance"),
    ("MOTILALOFS", "NSE_EQ|INE338A01010", "Finance"),
    ("ANGELONE",   "NSE_EQ|INE732I01013", "Finance"),
    ("HDFCAMC",    "NSE_EQ|INE127D01025", "Finance"),   # Nifty 500 / F&O
    ("CAMS",       "NSE_EQ|INE596I01012", "Finance"),   # Nifty 500 high vol
    ("CDSL",       "NSE_EQ|INE736A01011", "Finance"),   # Nifty 500 / F&O
    ("MCX",        "NSE_EQ|INE745G01035", "Finance"),   # Nifty 500 / F&O
    ("BSE",        "NSE_EQ|INE118H01025", "Finance"),   # Nifty 500
    ("MFSL",       "NSE_EQ|INE916I01018", "Finance"),   # Max Financial Nifty500
    ("PNBHOUSING", "NSE_EQ|INE572E01012", "Finance"),   # Nifty 500 / F&O

    # ─── CONSUMER (26) ────────────────────────────────────────────────────────
    ("TITAN",      "NSE_EQ|INE280A01028", "Consumer"),
    ("TRENT",      "NSE_EQ|INE849A01020", "Consumer"),
    ("PAGEIND",    "NSE_EQ|INE761H01022", "Consumer"),
    ("VOLTAS",     "NSE_EQ|INE226A01021", "Consumer"),
    ("CROMPTON",   "NSE_EQ|INE299U01018", "Consumer"),
    ("WHIRLPOOL",  "NSE_EQ|INE716A01013", "Consumer"),
    ("BATAINDIA",  "NSE_EQ|INE176A01028", "Consumer"),
    ("RELAXO",     "NSE_EQ|INE131B01039", "Consumer"),
    ("INDHOTEL",   "NSE_EQ|INE053A01029", "Consumer"),
    ("NAUKRI",     "NSE_EQ|INE663F01032", "Consumer"),
    ("ZOMATO",     "NSE_EQ|INE758T01015", "Consumer"),
    ("POLICYBZR",  "NSE_EQ|INE417T01026", "Consumer"),
    ("DMART",      "NSE_EQ|INE192R01011", "Consumer"),
    ("ASIANPAINT", "NSE_EQ|INE021A01026", "Consumer"),
    ("BERGEPAINT", "NSE_EQ|INE463A01038", "Consumer"),
    ("PIDILITIND", "NSE_EQ|INE318A01026", "Consumer"),
    ("SHOPERSTOP", "NSE_EQ|INE498B01024", "Consumer"),
    ("JUBLFOOD",   "NSE_EQ|INE797F01020", "Consumer"),
    ("DEVYANI",    "NSE_EQ|INE741K01010", "Consumer"),
    ("KAYNES",     "NSE_EQ|INE918Z01012", "Consumer"),
    ("PAYTM",      "NSE_EQ|INE982J01020", "Consumer"),
    ("LICI",       "NSE_EQ|INE0J1Y01017", "Consumer"),
    ("VEDANTFASH", "NSE_EQ|INE825R01014", "Consumer"),  # Nifty 500 / F&O Manyavar
    ("NYKAA",      "NSE_EQ|INE415W01015", "Consumer"),  # Nifty 500 / F&O
    ("METROBRAND", "NSE_EQ|INE657R01010", "Consumer"),  # Nifty 500
    ("EIH",        "NSE_EQ|INE230A01023", "Consumer"),  # Oberoi Hotels Nifty500

    # ─── EXTRA NIFTY 500 TOP-UPS (20 to reach 250) ───────────────────────────
    # Chemicals (high-liquidity, Nifty 500 / F&O)
    ("SRF",        "NSE_EQ|INE647A01010", "Metals"),     # Srf Ltd  – F&O
    ("DEEPAKNTR",  "NSE_EQ|INE288B01029", "Metals"),     # Deepak Nitrite – F&O
    ("AARTIIND",   "NSE_EQ|INE769A01020", "Metals"),     # Aarti Industries F&O

    # Finance top-ups
    ("LTFH",       "NSE_EQ|INE498L01015", "Finance"),    # L&T Finance – F&O
    ("KFINTECH",   "NSE_EQ|INE138Y01010", "Finance"),    # KFin Tech – Nifty 500

    # Consumer / Retail top-ups
    ("GODREJIND",  "NSE_EQ|INE233A01035", "Consumer"),  # Godrej Industries – F&O
    ("DELHIVERY",  "NSE_EQ|INE148U01020", "Consumer"),  # Delhivery – Nifty 500
    ("HONAUT",     "NSE_EQ|INE671A01010", "Consumer"),  # Honeywell Automation – Nifty500

    # Infrastructure / Capital Goods top-ups
    ("BHARAT",     "NSE_EQ|INE647001018", "Infrastructure"),  # Bharat Dynamics – F&O
    ("ELECON",     "NSE_EQ|INE225A01022", "Infrastructure"),  # Elecon Eng – Nifty 500
    ("HUDCO",      "NSE_EQ|INE031A01017", "Infrastructure"),  # HUDCO – Nifty 500

    # Energy top-ups
    ("GUJGASLTD",  "NSE_EQ|INE844O01030", "Energy"),    # Gujarat Gas – F&O Nifty500
    ("MGL",        "NSE_EQ|INE890D01026", "Energy"),     # Mahanagar Gas – F&O
    ("IGL",        "NSE_EQ|INE203G01027", "Energy"),     # Indraprastha Gas – F&O
    ("ATGL",       "NSE_EQ|INE399L01023", "Energy"),     # Adani Total Gas – F&O

    # Pharma top-up
    ("PFIZER",     "NSE_EQ|INE003B01020", "Pharma"),    # Pfizer India – Nifty 500
    ("SANOFI",     "NSE_EQ|INE058A01010", "Pharma"),    # Sanofi India – Nifty 500

    # IT top-up
    ("INTELLECT",  "NSE_EQ|INE306R01017", "IT"),         # Intellect Design – Nifty500

    # Banking top-up
    ("UJJIVANSFB", "NSE_EQ|INE334L01012", "Banking"),   # Ujjivan SFB – Nifty 500
    ("EQUITASBNK", "NSE_EQ|INE063B01027", "Banking"),   # Equitas SFB – Nifty 500
]

# Sector indices (unchanged)
INDICES = [
    ("NIFTY ENERGY",      "NSE_INDEX|Nifty Energy",      "Energy"),
    ("NIFTY BANK",        "NSE_INDEX|Nifty Bank",        "Banking"),
    ("NIFTY IT",          "NSE_INDEX|Nifty IT",          "IT"),
    ("NIFTY FMCG",        "NSE_INDEX|Nifty FMCG",        "FMCG"),
    ("NIFTY PHARMA",      "NSE_INDEX|Nifty Pharma",      "Pharma"),
    ("NIFTY AUTO",        "NSE_INDEX|Nifty Auto",        "Auto"),
    ("NIFTY METAL",       "NSE_INDEX|Nifty Metal",       "Metals"),
    ("NIFTY FIN SERVICE", "NSE_INDEX|Nifty MS Fin Serv", "Finance"),
    ("NIFTY INFRA",       "NSE_INDEX|Nifty Infra",       "Infrastructure"),
    ("NIFTY CONSUMPTION", "NSE_INDEX|Nifty Consumption", "Consumer"),
    ("NIFTY MULTI INFRA", "NSE_INDEX|Nifty Multi Infra", "Cement"),
    ("NIFTY SERV SECTOR", "NSE_INDEX|Nifty Serv Sector", "Telecom"),
]

# Build and write
rows = []
for sym, tok, sec in EQUITIES:
    rows.append({"symbol": sym, "instrument_token": tok, "sector": sec, "is_index": False})
for sym, tok, sec in INDICES:
    rows.append({"symbol": sym, "instrument_token": tok, "sector": sec, "is_index": True})

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False, lineterminator="\n")

equities_df = df[df["is_index"] == False]
indices_df  = df[df["is_index"] == True]
print(f"Written to    : {OUTPUT}")
print(f"Equity stocks : {len(equities_df)}")
print(f"Sector indices: {len(indices_df)}")
print(f"Total rows    : {len(df)}")
print()
print("Breakdown by sector:")
print(equities_df.groupby("sector")["symbol"].count().sort_values(ascending=False).to_string())
