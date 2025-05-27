const posts = [
    { id : 1, title : 'Post One' },
    { id : 1, title : 'Post One'}
];

// exporting 3 different ways:

    // export const getPosts = () => posts;

    // export { getPosts };
    
    const getPosts = () => posts;
    
    export default getPosts;

export const getPostsLength = () => posts.length;