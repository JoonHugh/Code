let posts = [
    { id: 1, title: "Blog post 1" },
    { id: 2, title: "Blog post 2" },
    { id: 3, title: "Blog post 3" }
];

// @desc Get all posts
// @route GET /api/posts
export const getPosts = (req, res) => {
    const limit = parseInt(req.query.limit);

    if (!isNaN(limit) && limit > 0) {
        return res.status(200).json(posts.slice(0, limit));
    } // if
    res.status(200).json(posts);
    
    console.log(req.query);
} // getPosts

// @desc Get single post
// @route GET /api/posts/:id
export const getPost = (req, res, next) => {
    // console.log(req.params.id);
    const id = parseInt(req.params.id);
    // res.status(200).json(posts.filter(post => post.id === id));
    const post = posts.find((post) => post.id === id);
    if (!post) {
        const error = new Error(`post with id ${id} doesn't exist!`);
        error.status = 404;
        return next(error);
    } // if
    
    res.status(200).json(post); 
} // getPost

// @desc Create new post
// @route POST /api/posts
export const createPost = (req, res, next) => {
    const newPost = {
        id: posts.length + 1,
        title: req.body.title
    }
    // console.log("newPost:", newPost);

    if (!newPost.title) {
        // return res.status(400).json({ msg: "Please include a title" });
        const error = new Error("Please include a title")
        error.status = 400;
        return next(error);
    } // if
    
    posts.push(newPost);
    res.status(201).json(posts);
} // postPost

// @desc update post
// @route PUT /api/posts/:id
export const putPost = (req, res, next) => {
    const id = parseInt(req.params.id);
    const post = posts.find((post) => post.id === id);
    
    if (!post) {
        const error = new Error(`No post with id ${id} was found!`);
        error.status = 404;
        return next(error);
    } // if
    
    post.title = req.body.title;
    res.status(200).json(posts);
} // putPost

// @desc Delete post
// @route DELETE /api/posts/:id
export const deletePost = (req, res, next) => {
    const id = parseInt(req.params.id);
    const post = posts.find((post) => post.id === id);

    if (!post) {
        const error = new Error(`Post with id ${id} was not found. Please enter a valid post`);
        error.status = 404;
        next(error);
    } // if
    posts = posts.filter((post => post.id !== id));

    res.status(200).json(posts);
} // deletePost

