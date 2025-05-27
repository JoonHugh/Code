// import { getPosts } from './postController.js'
import getPosts, { getPostsLength } from './postController.js';

const DEBUG = 0; 
// can use node index.js || node index || node .


// no window or document objects
if (DEBUG) {
    console.log("Hello world!");
    console.log(global);
    console.log(process);
}

// const { generateRandomNumber, celciusToFarenheit } = require('./utils');

// console.log(`Random Number: ${generateRandomNumber()}`);
// console.log(`Celcius to Farenheit: ${celciusToFarenheit(23)}`);

console.log(getPosts());
console.log(`Post Length: ${getPostsLength()}`);

