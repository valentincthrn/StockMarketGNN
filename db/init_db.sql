CREATE TABLE IF NOT EXISTS stocks_metadata (
    symbol TEXT PRIMARY KEY,
    name TEXT NOT NULL, 
    industry TEXT,
    sector TEXT,
    business_summary TEXT
);

CREATE TABLE IF NOT EXISTS stocks (
    symbol TEXT NOT NULL,
    quote_date TEXT NOT NULL,
    open REAL NOT NULL,
    close REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, quote_date),
    FOREIGN KEY (symbol) references stocks_metadata (symbol)
);

CREATE TABLE IF NOT EXISTS macro (
    indicators TEXT NOT NULL,
    quote_date DATE NOT NULL,
    valor REAL,
    PRIMARY KEY (indicators, quote_date)
);


