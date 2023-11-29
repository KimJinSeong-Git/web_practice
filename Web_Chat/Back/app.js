const express = require('express');
const router = express.Router();
const customerRoute = require('./routes/customer');
const productRoute = require('./routes/product');
const app = express();
const port = 3000;

app.use(express.json({
    limit: '50mb'
}));

app.listen(port, () => {
    console.log(`서버 실행. http://localhost:${port}`);
});

app.use('/customer', customerRoute);
app.use('/product', productRoute);