import styled from 'styled-components';

const Wrapper = styled.div`
    max-width: 95vw;
    height:100vh;
    margin: 0 auto;
`

function Container({ children }) {


    return(
        <Wrapper>
            {children}
        </Wrapper>
    );

}

export default Container