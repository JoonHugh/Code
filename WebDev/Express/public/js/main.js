const output = document.querySelector('#output');
const button = document.querySelector('#get-posts-btn');
const form = document.querySelector('#add-post-form');

async function showPosts() {
    try {
        const res = await fetch('http://localhost:5000/api/posts')
        if (!res.ok) {
            throw new Error('Failed to fetch posts');
        }
        
        const posts = await res.json();
        output.innerHTML = '';
        
        posts.forEach((post) => {
            const postElem = document.createElement('div');
            postElem.textContent = post.title;
            output.appendChild(postElem);
        });
    } catch (error) {
        console.log('Error fetching post: ', error);
    }
}

// Submit new Post
async function addPost(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const title = formData.get('title');

    try {
        const res = await fetch('http://localhost:5000/api/posts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({title})
        });

        if (!res.ok) {
            throw new Error('Failed to create new post');
        }

        const newPost = await res.json();
        const postElem = document.createElement('div');
        postElem.textContent = newPost.title;
        output.appendChild(postElem);
        showPosts();
    } catch (error) {
        console.log('Error creating post: ', error);
    }
}

// event listeners
button.addEventListener('click', showPosts);
form.addEventListener('submit', addPost);
