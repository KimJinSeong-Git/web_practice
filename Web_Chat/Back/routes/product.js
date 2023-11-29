const express = require('express');
const router = express.Router();

router
    .get('/', (req, res) => {
        res.send('상품 정보 조회');
    })
    .post('/insert', (req, res) => {
        res.send('신규 상품 추가');
    })
    .put('/update', (req, res) => {
        res.send('상품 정보 수정');
    })
    .delete('/delete', (req, res) => {
        res.send('상품 정보 삭제');
    });

module.exports = router;