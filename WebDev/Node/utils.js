function generateRandomNumber() {
    return Math.floor(Math.random() * 100 + 1)
}

// default export
module.exports = generateRandomNumber;// exporting function, import it somewhere else 
// You can also export noot just functions but other data such as objects 


function celciusToFarenheit(celcius) {
    return (celcius * 9) / 5 + 32
}

module.exports = {
    generateRandomNumber,
    celciusToFarenheit
};