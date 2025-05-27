import { EventEmitter } from 'events';

const myEmitter = new EventEmitter();

function greetHandler(name) {
    console.log('Hello World', name);
}

function goodbyeHandler(name) {
    console.log('Goodbye World', name);
}

// register event listeners 
myEmitter.on('greet', greetHandler);
myEmitter.on('goodbye', goodbyeHandler);


// Emit events
myEmitter.emit('greet', 'John');
myEmitter.emit('goodbye', 'Joon');

// Error handling
myEmitter.on('error', (err) => {
    console.log("an error occured", err);
});

myEmitter.emit('error', new Error('Something went wrong'));

