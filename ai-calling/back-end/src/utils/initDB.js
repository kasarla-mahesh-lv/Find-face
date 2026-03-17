
const pool = require('../config/db');

async function initDB() {
  try {
    const query = `
      CREATE TABLE IF NOT EXISTS calls(
        id SERIAL PRIMARY KEY,
        phone_number VARCHAR(20),
        digit_pressed INT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    `;
    
    await pool.query(query);

    const seedRows = [
      ["+15551234567", 1],
      ["+15557654321", 2],
      ["+15559876543", 3],
      ["+15553456789", 6],
    ];

    for (const [phone, digit] of seedRows) {
      await pool.query(
        `INSERT INTO calls (phone_number, digit_pressed)
         SELECT $1::varchar(20), $2::int
         WHERE NOT EXISTS (
           SELECT 1
           FROM calls
           WHERE phone_number = $1::varchar(20) AND digit_pressed = $2::int
         )`,
        [phone, digit]
      );
    }
    
    console.log("Database table ready");
    
  } catch (error) {
    console.error("Database initialization error:", error);
    throw error;
  }
}

module.exports = initDB;
