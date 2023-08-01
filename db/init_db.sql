CREATE TABLE IF NOT EXISTS stocks_metadata (
    symbol TEXT PRIMARY KEY,
    industry TEXT NOT NULL,
    sector TEXT NOT NULL,
    business_summary TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stocks (
    symbol TEXT NOT NULL,
    quote_date TEXT NOT NULL,
    open REAL NOT NULL,
    close REAL NOT NULL,
    high REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (symbol, quote_date),
    FOREIGN KEY (symbol) references stocks_metadata (symbol)
);

