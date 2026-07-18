# Interface de demonstração manual do CensuraBot, separada da API real
# (api.py). Serve pra testar/mostrar o resultado das camadas de moderação
# sem precisar montar uma requisição HTTP - roda com `streamlit run app.py`.
import streamlit as st

from moderador import censurar_mensagem

st.set_page_config(page_title="CensuraBot", page_icon=":speech_balloon:")  

st.title("CensuraBot")
st.write("Digite uma mensagem, detecte o termo ofensivo e receba a mensagem censurada.")

mensagem = st.text_area("Mensagem", placeholder="Digite sua mensagem:")

if st.button("Analisar:"):
    if not mensagem.strip():
        st.warning("Digite uma mensagem.")
    else:
        mensagem_censurada = censurar_mensagem(mensagem)

        st.subheader("Mensagem Censurada:")
        st.write(mensagem_censurada)

        if mensagem != mensagem_censurada:
            st.warning("Atenção: A mensagem contém termos ofensivos e foi censurada.")
        else:
            st.success("A mensagem não contém termos ofensivos.")