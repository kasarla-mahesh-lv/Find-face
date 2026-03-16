const pool = require("../config/db")

async function saveCall(phone, digit) {

  const query =
  "INSERT INTO calls(phone_number, digit_pressed) VALUES($1,$2)"

  await pool.query(query,[phone,digit])

}

module.exports = { saveCall }