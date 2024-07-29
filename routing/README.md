<a href="https://sambanova.ai/">
<picture>
 <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Routing
======================
This is an example of routing a user query to different RAG pipeline or LLM 
based on keywords from the datasource.

<!-- TOC -->

- [Before you begin](#before-you-begin)
    - [Clone this repository](#clone-this-repository)
    - [Set up the account and config file for the LLM](#set-up-the-account-and-config-file-for-the-llm)
        - [Setup for SambaStudio users](#setup-for-sambastudio-users)
        - [Setup for Sambaverse users](#setup-for-sambaverse-users)
        - [Setup for FasCoE users](#setup-for-fascoe-users)
        - [Install dependencies](#install-dependencies)
- [Use the Routing](#use-the-routing)
    - [Quick start](#quick-start)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->

# Before you begin

To use this in your application you need an instruction model, we recommend to use the Mistral 7B Instruct or Meta Llama3 8B, either from Sambaverse or from SambaStudio CoE. For embedding model, we recommend to use intfloat/e5-large-v2.

## Clone this repository

Clone the starter kit repo.
```
git clone https://github.com/sambanova/ai-starter-kit.git
```

## Set up the account and config file for the LLM 

The next step sets you up to use one of the models available from SambaNova. It depends on whether you're a SambaNova customer who uses SambaStudio, FastCoE endpoint or you want to use the publicly available Sambaverse.

### Setup for SambaStudio users

To perform this setup, you must be a SambaNova customer with a SambaStudio account.

1. Log in to SambaStudio and get your API authorization key. The steps for getting this key are described [here](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html#_acquire_the_api_key).
2. Select the model you want to use (e.g. CoE containing Meta-Llama-Guard-2-8B) and deploy an endpoint for inference. See the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).
3. In the repo root directory create an env file in  `sn-ai-starter-kit/.env`, and update it with your Sambastudio endpoint variables ([view your endpoint information](https://docs.sambanova.ai/sambastudio/latest/endpoints.html#_view_endpoint_information)), Here's an example:

    - Assume you have an endpoint with the URL
        "https://api-stage.sambanova.net/api/predict/generic/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"

    - You can enter the following in the env file (with no spaces):

    ``` bash
    SAMBASTUDIO_BASE_URL="https://api-stage.sambanova.net"
    SAMBASTUDIO_BASE_URI="api/predict/generic"
    SAMBASTUDIO_PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
    SAMBASTUDIO_ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
    SAMBASTUDIO_API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
    ```

4. Open the [config file](./config.yaml), in `llm` section set the variable `api` to `"sambastudio"`, and set the `sambaverse_model_name`, `coe` and `select_expert` configs and save the file.

### Setup for Sambaverse users 

1. Create a Sambaverse account at [Sambaverse](sambaverse.sambanova.net) and select your model. 
2. Get your [Sambaverse API key](https://docs.sambanova.ai/sambaverse/latest/use-sambaverse.html#_your_api_key) (from the user button).
3. In the repo root directory create an env file in `sn-ai-starter-kit/.env` and specify the Sambaverse API key (with no spaces), as in the following example:

    ``` bash
        SAMBAVERSE_API_KEY="456789ab-cdef-0123-4567-89abcdef0123"
    ```

4. In the [config file](./config.yaml), in `llm` section set the `api` variable to `"sambaverse"`, and set the `sambaverse_model_name`  and `select_expert` configs.

### Setup for FasCoE users 

- In the repo root directory create an env file in `sn-ai-starter-kit/.env` and specify the FastCoE url and the FastCoE API key (with no spaces), as in the following example:

    ``` bash
        FAST_COE_URL = "https://abcd.snova.ai/api/v1/chat/completion"
        FAST_COE_API_KEY = "456789abcdef0123456789abcdef0123"
    ```

- In the [config file](./config.yaml), in `llm` section set the `api` variable to `"fastcoe"`, and set the `select_expert` config.

###  Install dependencies

We recommend that you run the starter kit in a virtual environment.

NOTE: python 3.10 or higher is required to use this kit.

1. Install the python dependencies in your project environment.

    ```bash
    cd ai_starter_kit/routing
    python3 -m venv routing_env
    source routing_env/bin/activate
    pip  install  -r  requirements.txt
    ```

# Use the Routing

## Quick start

We provide a simple module for using the Routing, for this you will need:

1. Extract keywords from datasource:

    You should create keywords from your datasource and save them in the local.

    The class and main functions are in [keyword_extraction_custom_keyllm.py](keyword_extraction_custom_keyllm.py).

2. Load keywords and pass it to the prompt. An example is in [routing.py](routing.py).


# Third-party tools and data sources

All the packages/tools are listed in the `requirements.txt` file in the project directory. Some of the main packages are listed below:

* langchain (version 0.2.8)
* python-dotenv (version 1.0.1)
* langchain_community (version 0.2.7)
* langchain_core (version 0.2.19)
* torch (version 2.1.1)
* keybert (version 0.8.5)
* keyphrase_vectorizers (version 0.0.13)