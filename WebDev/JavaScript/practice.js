/*
    What is Javascript?
    * High level, interpreted programming language (a scripting language)
    * Conforms to the ECMAScript specification
    * Multi-paradigm
    * Runs on the client/browser as well as on the server (Node.js) - a JavaScript runtime
    * Frontend stuff using forms and buttons etc.

    Why learn Javascript?
    * It is the programming language of the browser (client side programming - programs that run on your machine)
    * Build very interactive user interfaces with frameworks like React
    * Used in building very fast server side and full stack applications
    * Used in mobile development (React Native, NativeScript, Ionic)
    * Used in desktoop application development (Electron JS)
    

    What you will learn:
    1. Variable & Data types
    2. Arrays
    3. Object Literals
    4. Methods for strings, arrays, objects, etc
    5. Loops - for, while, for...of, forEach, map
    6. Conditionals (if, ternary & switch)
    7. Functions (normal & arrow)
    8. OOP (prototypes and classes)
    9. DOM selection
    10. DOM manipulation
    11. Events
    12. Basic Form Validation
*/

/*

console.log("Hello World"); 
console.error("This is an error");
console.warn("This is a warning");

*/

/*
3 different variables:
var: var is globally scoped. If you have a if statement and there's a variable, and there's another variable with that same name outside of that if statement, there can be a conflict
let: reassign values
const: cannot reassign. Always use const unless you knoow you're going to reassign the value
*/

let age = 30;
age = 31;

console.log(age)

/*
const name = "Joon";
name = "Aditiya";
console.log(name); // error because constant */

let score;
score = 10;

console.log(score);

/*
data types:
Strings -
Numbers -
Boolean -
null -
undefined -
Symbol -
*/

const name = "Joon";
const height = 170;
const rating = 4.5;
const isCool = true;
const x = null; // empty a var but nothing in it
const y = undefined;
let z; // same as undefined

console.log(typeof y); // typeof x = object. bogus value.


// concatenation
console.log("My name is" + name + "and I am age" + age); // old way

const hello = `My name is ${name} and I am ${age}`;

console.log(hello);


const s = `Helo World`;

console.log(s.length);
console.log(s.toUpperCase())
console.log(s.toLowerCase())

console.log(s.substring(0,5).toUpperCase());

console.log(s.split(' '))

const ex = 'technology, computers, it, code';
console.log(ex.split(', '));


// Arrays 

/*
 Multi-line comments
 Variables that hold multiple values
*/

const numbers = [1, 2, 3, 4, 5];
console.log(numbers);

const fruits = ["apples", "oranges", "pears", 10, false, true, 1.8]
console.log(fruits);


for (let i = 0; i < fruits.length; i++) {
    console.log(fruits[i]);
}
    
console.log(fruits[0]);

fruits.push("grapes");

console.log(fruits)

fruits.unshift("watermelons")

console.log(fruits);

console.log(fruits.pop())

console.log(fruits);

console.log(Array.isArray(fruits));
console.log(Array.isArray(s));

const array2 = ["pie", "cake", "ice cream"]
newArray = array2.concat(fruits)
console.log(newArray);
console.log(fruits.indexOf("oranges")) // returns index of elements in arrays
// .push() appends, .unshift() prepends .pop() pops from end of array

const person = {
    firstName: "Joon",
    lastName: "Hugh",
    age:30,
    hobbies: ["Tennis", "Golf", "Video Games"],
    address: {
        street: "937 King Horn Ct",
        city: "Buford",
        state: "Georgia"
    }
}

console.log(person);
console.log(person.firstName, person.lastName);
console.log(person.hobbies[1]);
console.log(person.address.city);

// destructuring
const { firstName, lastName, address:{ city }} = person; // pulling these out of the person object
console.log(firstName, lastName, city)

person.email = "joonhugh2021@gmail.com";

console.log(person)


const todos = [
    {
        id: 1,
        text: 'Take out trash',
        isCompleted: true
    },
    {
        id: 2,
        text: 'finish JS crash course video',
        isCompleted: false
    },
    {
        id: 3,
        text: 'pickup hyoyi',
        isCompleted: false
    }

];

console.log(todos[1].text);

// JSON

const todoJSON = JSON.stringify(todos)

console.log(todoJSON);

// for loop
for(let i = 0; i < 10; i++) {
    console.log(`for loop numer: ${i}`);
}

let i = 0;
while (i < 10) {
    console.log(`while loop practice: ${i}`)
    i++ // important lmao
}


for (let i = 0; i < todos.length; i++) {
    console.log(`todo[${i}]:`, todos[i])
} // for

// EASIER
for (let todo of todos) {
    console.log(todo);
}

// high-order arrays methods
// forEach, map, filter
// forEach
todos.forEach(function(todo) {
    console.log(todo);
});

// map
const todoText = todos.map(function(todo) {
    return todo.text;
});

console.log(todoText);

// filter
const  todoCompleted = todos.filter(function(todo) {
    return todo.isCompleted == true;
});

console.log(todoCompleted);

// putting together filter and map
const todoCompletedText = todos.filter(function(todo) {
    return todo.isCompleted === true;
}).map(function(todo) {
    return todo.text
});

console.log(todoCompletedText)

// conditionals
const num = `10`;
if (num == 10) { // "==" only checks if it's equal, doesn't have to specify data type
    console.log("true, x = 10") 
}

const num2 = "21";
if (num2 === 10) {
    console.log("Hello WOrld");
} else if (num2 > 10) {
    console.log("greater than 10");
} else {
    console.log("less than 10");
}

if (num > 11 || num2 > 20) {
    console.log("num is greater than 11 OR num2 is greater than 20");
}
if (num > 9 && num2 > 20) {
    console.log("num is greater than 11 AND num2 is greater than 20");
}

// ternary operator
const p = 11;
const color = p > 10 ? 'red' : 'blue';
console.log(color);

switch(color) {
    case 'red': 
        console.log("color is red");
        break;
    case 'blue': 
        console.log("color is blue");
        break;
    default: 
        console.log("color is not a red or blue");
        break;
}

function addNums(num1, num2) {
    console.log(num1 + num2)
} // addNums

addNums(5, 4);
addNums();

const addNumsTogether = (num1, num2) => num1 + num2
// function name = input => output

console.log(addNumsTogether(5, 5));

// OOP
/*
function Person(firstName, lastName, dob) {
    this.firstName = firstName;
    this.lastName = lastName;
    this.dob = new Date(dob);

    /*this.getBirthYear = function() {
        return this.dob.getFullYear();
    }

    this.getFullName = function() {
        return `${this.firstName} ${this.lastName}`;
    }*
} // Person

Person.prototype.getBirthYear = function() {
    return this.dob.getFullYear();
}

Person.prototype.getFullName = function() {
    return `${this.firstName} ${this.lastName}`;
}*/


// Class
class Person {
    constructor(firstName, lastName, dob) {
        this.firstName = firstName;
        this.lastName = lastName;
        this.dob = new Date(dob);
    }

    getBirthYear() {
        return this.dob.getFullYear();
    }

    getFullName() {
        return `${this.firstName} ${this.lastName}`;
    }
}

// Instantiate person object
const person1 = new Person("Joon", "Hugh", "07-03-2003");
const person2 = new Person("Aditiya", "Patel", "05-23-2003");
console.log(person1);
console.log(person1.dob);
console.log(person1.getBirthYear());
console.log(person1.getFullName());

console.log(person2.getFullName());





// DOM
