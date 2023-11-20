DELETE FROM stocks
        WHERE rowid IN (
        SELECT rowid FROM stocks
        WHERE symbol = ?
        ORDER BY quote_date DESC
        LIMIT 3
        );