{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "jN5logZdbGlN"
      },
      "outputs": [],
      "source": [
        "# This code is designed to run as an AWS Lambda function that:\n",
        "\n",
        "# Takes a blog topic from an HTTP request (via API Gateway).\n",
        "\n",
        "# Uses Amazon Bedrock (AI model) to generate a 200-word blog.\n",
        "\n",
        "# Saves the blog to an S3 bucket named repto."
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import boto3\n",
        "import botocore.config\n",
        "import json\n",
        "from datetime import datetime"
      ],
      "metadata": {
        "id": "u3O0ppadbaXx"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# boto3: AWS SDK for Python – used to interact with Bedrock and S3 services.\n",
        "\n",
        "# botocore.config: Allows you to configure settings like timeout and retries for boto3 clients.\n",
        "\n",
        "# json: For parsing and formatting JSON data (event input/output).\n",
        "\n",
        "# datetime: Used to timestamp the blog file when saving it to S3."
      ],
      "metadata": {
        "id": "1Ptcn7Aqbdcg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Function 1: blog_generate_using_bedrock(blogtopic)"
      ],
      "metadata": {
        "id": "AiFh0fCXbdic"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# This function:\n",
        "\n",
        "# Connects to Amazon Bedrock.\n",
        "\n",
        "# Requests a blog using the Titan Text model.\n",
        "\n",
        "# Returns the generated text."
      ],
      "metadata": {
        "id": "oze96hAMbdkk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 1: Create the prompt and config"
      ],
      "metadata": {
        "id": "aZBUvNGDbdnP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "body = {\n",
        "    \"inputText\": f\"Write a 200-word blog on the topic: {blogtopic}\",\n",
        "    \"textGenerationConfig\": {\n",
        "        \"maxTokenCount\": 512,\n",
        "        \"stopSequences\": [],\n",
        "        \"temperature\": 0.5,\n",
        "        \"topP\": 0.9\n",
        "    }\n",
        "}"
      ],
      "metadata": {
        "id": "d_XegIz0bdps"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# This is the input for the Bedrock model.\n",
        "\n",
        "# Prompt: What the model should generate.\n",
        "\n",
        "# Temperature: Controls randomness (0.5 is moderately creative).\n",
        "\n",
        "# topP: Another sampling control method.\n",
        "\n",
        "# maxTokenCount: Limit on length of output."
      ],
      "metadata": {
        "id": "FO8UzSIgbdr3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#  Step 2: Initialize Bedrock client"
      ],
      "metadata": {
        "id": "ZIcRjxsAbdue"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "bedrock = boto3.client(\n",
        "    \"bedrock-runtime\",\n",
        "    region_name=\"ap-south-1\",\n",
        "    config=botocore.config.Config(read_timeout=300, retries={\"max_attempts\": 3})\n",
        ")"
      ],
      "metadata": {
        "id": "sadRNMEFbdw_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Creates a connection to Amazon Bedrock runtime in the Mumbai (ap-south-1) region.\n",
        "\n",
        "# Timeout: 300 seconds, Retry: 3 times if it fails."
      ],
      "metadata": {
        "id": "U5foUr7YbdzW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 3: Call the Titan model"
      ],
      "metadata": {
        "id": "TVbQP3RObd2B"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "response = bedrock.invoke_model(\n",
        "    body=json.dumps(body),\n",
        "    modelId=\"amazon.titan-text-express-v1\",\n",
        "    contentType=\"application/json\",\n",
        "    accept=\"application/json\"\n",
        ")"
      ],
      "metadata": {
        "id": "k63Gi6gJbd4a"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# This calls the model to generate the blog content.\n",
        "\n",
        "# modelId=\"amazon.titan-text-express-v1\" specifies Titan model.\n",
        "\n",
        "# Input and output are in JSON format."
      ],
      "metadata": {
        "id": "3jsDQEBIbd6v"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 4: Parse the response"
      ],
      "metadata": {
        "id": "S4jR36qvbd9k"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "response_content = response.get(\"body\").read()\n",
        "response_data = json.loads(response_content)"
      ],
      "metadata": {
        "id": "lOZIP76nbeAR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Extracts and parses the model's response into usable data."
      ],
      "metadata": {
        "id": "PZ8SgS87beDc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 5: Extract blog text"
      ],
      "metadata": {
        "id": "vQNLLHqxcj1K"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "results = response_data.get(\"results\", [])\n",
        "if results and \"outputText\" in results[0]:\n",
        "    return results[0][\"outputText\"]"
      ],
      "metadata": {
        "id": "UT-0F2hOcj4K"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Finds the generated blog text inside the results array."
      ],
      "metadata": {
        "id": "Srf80dv3cj6j"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Function 2: save_blog_details_s3()"
      ],
      "metadata": {
        "id": "eK4HKURpcj9b"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def save_blog_details_s3(s3_key: str, s3_bucket: str, generate_blog: str):"
      ],
      "metadata": {
        "id": "HqkQuhceckA1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Saves the generated blog to an S3 bucket."
      ],
      "metadata": {
        "id": "TVz04jVTc9Gu"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "s3 = boto3.client(\"s3\")\n",
        "s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=generate_blog)"
      ],
      "metadata": {
        "id": "4-nIR-CGdChQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Connects to Amazon S3.\n",
        "# Uploads the blog to your specified bucket using the path blog-output/{timestamp}.txt."
      ],
      "metadata": {
        "id": "iW0ujqg3dRSd"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Function 3: lambda_handler(event, context)"
      ],
      "metadata": {
        "id": "CErQFcs1dRUy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# def lambda_handler(event, context):"
      ],
      "metadata": {
        "id": "WaGcziPjdRXY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# This is the main entry point for the AWS Lambda function."
      ],
      "metadata": {
        "id": "jjpA6rfCdRaS"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 1: Parse input event"
      ],
      "metadata": {
        "id": "_5Lvpy6BdRdr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "event = json.loads(event[\"body\"])\n",
        "blogtopic = event.get(\"blog_topic\", \"\")"
      ],
      "metadata": {
        "id": "50Ncpq-1dc5M"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# The function expects a POST request with JSON body.\n",
        "\n",
        "# Extracts the topic string (\"blog_topic\").\n"
      ],
      "metadata": {
        "id": "e480Ek9_dc7h"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 2: Input validation"
      ],
      "metadata": {
        "id": "UlXzMbkEdc92"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "if not blogtopic:\n",
        "    return {\n",
        "        \"statusCode\": 400,\n",
        "        \"body\": json.dumps(\"Invalid input: 'blog_topic' is required.\")\n",
        "    }\n"
      ],
      "metadata": {
        "id": "-522YyNGddAM"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Returns error if no topic is given."
      ],
      "metadata": {
        "id": "hWFMtB8UddCh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 3: Generate blog"
      ],
      "metadata": {
        "id": "wwvlyKlZddE4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "generate_blog = blog_generate_using_bedrock(blogtopic=blogtopic)"
      ],
      "metadata": {
        "id": "7ptTOG3sddHh"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Calls Bedrock to generate the blog"
      ],
      "metadata": {
        "id": "pKYi658ZddK5"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 4: Save blog if successful"
      ],
      "metadata": {
        "id": "bcfD-07fdyKs"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "current_time = datetime.now().strftime(\"%Y%m%d%H%M%S\")\n",
        "s3_key = f\"blog-output/{current_time}.txt\"\n",
        "s3_bucket = \"repto\"\n",
        "save_blog_details_s3(s3_key, s3_bucket, generate_blog)"
      ],
      "metadata": {
        "id": "K8hflNLhdyM8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Creates a unique filename using timestamp.\n",
        "\n",
        "# Saves it to the bucket repto.\n"
      ],
      "metadata": {
        "id": "dGDPErJAdyPA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Step 5: Return success response"
      ],
      "metadata": {
        "id": "FpU-hR7MdyRa"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "return {\n",
        "    \"statusCode\": 200,\n",
        "    \"body\": json.dumps(\"Blog generation and saving to S3 completed successfully.\")\n",
        "}\n"
      ],
      "metadata": {
        "id": "R4-gR12TdyUA"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Returns a success status if all steps pass."
      ],
      "metadata": {
        "id": "8BjFd6fadyWT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "DNwbAGfqdyZ1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "wPddwU9tfFnr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "JY4TxoVxfFp_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "l8H8J1oSfFsb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "mYZyDMWOfFu8"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# # Amazon Bedrock\n",
        "\n",
        "# Amazon Bedrock is a fully managed service by AWS that enables you to build and scale generative AI applications\n",
        "# using foundation models (FMs) from leading AI model providers like:\n",
        "\n",
        "# Anthropic (Claude models)\n",
        "\n",
        "# AI21 Labs (Jurassic)\n",
        "\n",
        "# Stability AI (Stable Diffusion)\n",
        "\n",
        "# Cohere\n",
        "\n",
        "# Meta (via AWS) (Llama)\n",
        "\n",
        "# Amazon Titan (Amazon’s own models)\n",
        "\n",
        "# (1).Key Features of Amazon Bedrock:\n",
        "# No infrastructure management:\n",
        "# You don’t need to manage or provision GPUs/servers. AWS handles all the compute and scaling behind the scenes.\n",
        "\n",
        "# API Access to Multiple Models:\n",
        "# Access a variety of powerful models via a single unified API without having to sign up with each model provider separately.\n",
        "\n",
        "# Custom Model Fine-tuning:\n",
        "# You can customize base models with your own data using few-shot, zero-shot, or fine-tuning methods, without retraining from scratch.\n",
        "\n",
        "# Integrated with AWS services:\n",
        "# Works well with tools like Amazon SageMaker, Lambda, Step Functions, DynamoDB, S3, etc.\n",
        "\n",
        "# Secure and Private:\n",
        "# Your data stays within your AWS environment. It’s not used to train underlying models.\n",
        "\n",
        "# Supports Text and Image Generation:\n",
        "# You can generate text (like blogs, code, summaries) and images (via models like Stable Diffusion) using Bedrock.\n",
        "\n",
        "# (2). What Can You Do With Bedrock?\n",
        "# Build chatbots\n",
        "\n",
        "# Generate blogs, emails, or articles\n",
        "\n",
        "# Do text summarization or classification\n",
        "\n",
        "# Create image-based applications\n",
        "\n",
        "# Develop virtual assistants\n",
        "\n",
        "# Use it for code generation\n",
        "\n",
        "# Apply it in search and recommendation systems"
      ],
      "metadata": {
        "id": "PyjBrWTLfFxF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# What Are Foundation Models (FMs) in Amazon Bedrock?\n",
        "# Foundation Models (FMs) are large, pre-trained machine learning models that serve as the base for a\n",
        "# wide range of generative AI tasks like:\n",
        "\n",
        "# Text generation (e.g., essays, emails, summaries)\n",
        "\n",
        "# Image generation\n",
        "\n",
        "# Code generation\n",
        "\n",
        "# Search and recommendations\n",
        "\n",
        "# Chatbots and virtual agents\n",
        "\n",
        "# Data analysis and insights\n",
        "\n",
        "# In Amazon Bedrock, you can access these foundation models via API—without having to build or train them from scratch."
      ],
      "metadata": {
        "id": "V1k-wGNffFzl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Provider -> Anthropic\n",
        "# Model Name -> Claude 2 / Claude 3\n",
        "# Type -> Text / Multi-modal\n",
        "# Best For -> Chatbots, document processing, reasoning\n",
        "\n",
        "# Provider -> AI21 Labs\n",
        "# Model Name -> Jurassic-2 (Mid, Ultra)\n",
        "# Type -> Text\n",
        "# Best For -> Long-form generation, code, Q&A\n",
        "\n",
        "# Provider -> Cohere\n",
        "# Model Name -> Command R / R+\n",
        "# Type -> Text\n",
        "# Best For -> RAG (Retrieval-Augmented Generation), search\n",
        "\n",
        "# Provider -> Stability AI\n",
        "# Model Name -> Stable Diffusion\n",
        "# Type -> Image\n",
        "# Best For -> Generating images from text\n",
        "\n",
        "# Provider -> Amazon\n",
        "# Model Name -> Titan Text, Titan Embeddings\n",
        "# Type -> Text, Embedding\n",
        "# Best For -> Summarization, classification, search\n",
        "\n",
        "# Provider -> Meta (via AWS)\n",
        "# Model Name -> Llama 2 / Llama 3\n",
        "# Type -> Text\n",
        "# Best For -> Open-weight, fine-tunable models\n"
      ],
      "metadata": {
        "id": "ysECKp61fF2O"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "q_ftPQNhfF5p"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "IQZW4TC3ho4i"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "ClxSOoUkho7T"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "5u7d_aiaho9_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Agenda ---->\n",
        "# Bedrock ---> Foundation Model (According to our purpose) ----> model-ID  , parameters\n",
        "# Lambda function ----> function(user_topic) ----> text generate -----> Output ----> s3 bucket\n",
        "# IAM ---> Role ---> All permissions\n",
        "\n",
        "# API GateWays ----> REST API  -----> Integrate with our Lambda function ."
      ],
      "metadata": {
        "id": "oAY8PibChpBD"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "Blog Generation Project -Step by Step Execution"
      ],
      "metadata": {
        "id": "ohvG4KTcwgSt"
      }
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "1BSsnW3HhpDc"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# step-1\n",
        "\n",
        "# Go to AWS ---> Bedrock ---> Now we will see 'Foundation Models' ---> click on 'llama3' .\n",
        "\n",
        "# Now we will see  Model access in the left side in Amazon bedrock  ---> for access granted click on 'Managed model access' and select checkbox of (AI21 , Jurassic-2 Ultra , Jurassic-2 Mid ,Emved English , Embed Multilingual, Command , Command light , Meta , Laama 3 88 Instruct , Lama 3 70B Instruct , Llama 2 Chat 13B , Llama2 Chat 70B , Llama2 13B , Llama2 70B , Stability AI , SDXL 1.0) ---> Now click to 'save' .\n",
        "# -------------------------------------\n",
        "\n",
        "# step-2 (Lambda)\n",
        "# Go to AWS ---> Lambda ---> click 'create function' --->select the checkbox  'Author from scratch'---> Fill these details :\n",
        "# Function name : awsappbedrock\n",
        "# Runtime : python 3.12\n",
        "\n",
        "# click on 'create function' .\n",
        "# click on configuration ---> click on 'Edit basic setting' --->\n",
        "# TimeOut[3]min [0]sec\n",
        "\n",
        "# click on 'save'\n",
        "# -------------------------------------\n",
        "\n",
        "# In Bedrock ---> Llama models --> select 'Llama2 chat 13B' --> copy the 'API request'and paste it into Lambda function .\n",
        "\n",
        "# Our model is (\n",
        "# Amazon Bedrock\n",
        "# Model catalog\n",
        "# Titan Text G1 - Express\n",
        "# Amazon Bedrock\n",
        "# Model catalog\n",
        "# Titan Text G1 - Express\n",
        "# amazon Logo\n",
        "# Titan Text G1 - Express\n",
        "# By: Amazon\n",
        "# | Access granted\n",
        "# Amazon Titan Text Express has a context length of up to 8,000 tokens, making it well-suited for a wide range of advanced, general language tasks such as open-ended text generation and conversational chat, as well as support within Retrieval Augmented Generation (RAG). At launch, the model is optimized for English, with multilingual support for more than 100 additional languages available in preview.)\n"
      ],
      "metadata": {
        "id": "dHB8iBI2hpGW"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# import boto3\n",
        "# import botocore.config\n",
        "# import json\n",
        "# from datetime import datetime\n",
        "\n",
        "# def blog_generate_using_bedrock(blogtopic: str) -> str:\n",
        "#     \"\"\"\n",
        "#     Generate a blog using Amazon Bedrock.\n",
        "#     \"\"\"\n",
        "#     body = {\n",
        "#         \"inputText\": f\"Write a 200-word blog on the topic: {blogtopic}\",\n",
        "#         \"textGenerationConfig\": {\n",
        "#             \"maxTokenCount\": 512,\n",
        "#             \"stopSequences\": [],\n",
        "#             \"temperature\": 0.5,\n",
        "#             \"topP\": 0.9\n",
        "#         }\n",
        "#     }\n",
        "\n",
        "#     try:\n",
        "#         # Initialize Bedrock client\n",
        "#         bedrock = boto3.client(\n",
        "#             \"bedrock-runtime\",\n",
        "#             region_name=\"ap-south-1\",\n",
        "#             config=botocore.config.Config(read_timeout=300, retries={\"max_attempts\": 3})\n",
        "#         )\n",
        "\n",
        "#         # Invoke the Bedrock model\n",
        "#         response = bedrock.invoke_model(\n",
        "#             body=json.dumps(body),\n",
        "#             modelId=\"amazon.titan-text-express-v1\",\n",
        "#             contentType=\"application/json\",\n",
        "#             accept=\"application/json\"\n",
        "#         )\n",
        "\n",
        "#         # Parse the response\n",
        "#         response_content = response.get(\"body\").read()\n",
        "#         response_data = json.loads(response_content)\n",
        "#         print(\"Bedrock response:\", response_data)\n",
        "\n",
        "#         # Extract the generated text\n",
        "#         results = response_data.get(\"results\", [])\n",
        "#         if results and \"outputText\" in results[0]:\n",
        "#             return results[0][\"outputText\"]\n",
        "\n",
        "#         return \"\"\n",
        "\n",
        "#     except Exception as e:\n",
        "#         print(f\"Error generating the blog: {e}\")\n",
        "#         return \"\"\n",
        "\n",
        "# def save_blog_details_s3(s3_key: str, s3_bucket: str, generate_blog: str):\n",
        "#     \"\"\"\n",
        "#     Save the generated blog to an S3 bucket.\n",
        "#     \"\"\"\n",
        "#     s3 = boto3.client(\"s3\")\n",
        "\n",
        "#     try:\n",
        "#         # Save the blog to S3\n",
        "#         s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=generate_blog)\n",
        "#         print(\"Blog saved to S3:\", s3_key)\n",
        "\n",
        "#     except Exception as e:\n",
        "#         print(f\"Error saving the blog to S3: {e}\")\n",
        "\n",
        "# def lambda_handler(event, context):\n",
        "#     \"\"\"\n",
        "#     AWS Lambda handler function.\n",
        "#     \"\"\"\n",
        "#     try:\n",
        "#         # Parse the input event\n",
        "#         event = json.loads(event[\"body\"])\n",
        "#         blogtopic = event.get(\"blog_topic\", \"\")\n",
        "\n",
        "#         if not blogtopic:\n",
        "#             return {\n",
        "#                 \"statusCode\": 400,\n",
        "#                 \"body\": json.dumps(\"Invalid input: 'blog_topic' is required.\")\n",
        "#             }\n",
        "\n",
        "#         # Generate the blog\n",
        "#         generate_blog = blog_generate_using_bedrock(blogtopic=blogtopic)\n",
        "\n",
        "#         if generate_blog:\n",
        "#             # Save the blog to S3\n",
        "#             current_time = datetime.now().strftime(\"%Y%m%d%H%M%S\")\n",
        "#             s3_key = f\"blog-output/{current_time}.txt\"\n",
        "#             s3_bucket = \"repto\"\n",
        "#             save_blog_details_s3(s3_key, s3_bucket, generate_blog)\n",
        "\n",
        "#             return {\n",
        "#                 \"statusCode\": 200,\n",
        "#                 \"body\": json.dumps(\"Blog generation and saving to S3 completed successfully.\")\n",
        "#             }\n",
        "#         else:\n",
        "#             return {\n",
        "#                 \"statusCode\": 500,\n",
        "#                 \"body\": json.dumps(\"Blog generation failed.\")\n",
        "#             }\n",
        "\n",
        "#     except Exception as e:\n",
        "#         print(f\"Error in lambda_handler: {e}\")\n",
        "#         return {\n",
        "#             \"statusCode\": 500,\n",
        "#             \"body\": json.dumps(f\"Internal server error: {e}\")\n",
        "#         }\n"
      ],
      "metadata": {
        "id": "4uqb83yjuw4o"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#  Now paste this lambda function into Our landa function in aws .\n",
        "\n",
        "# # Now click on 'deploy'\n",
        "\n",
        "# # In lambda function 'repto' is our s3 bucket name ."
      ],
      "metadata": {
        "id": "Lm0yMfbIu9if"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# For test in lambda function put this inn json\n",
        "# Test with Sample Event: Use the following event in the AWS Lambda test console:\n",
        "\n",
        "# {\n",
        "#   \"body\": \"{\\\"blog_topic\\\": \\\"Artificial Intelligence in Education\\\"}\"\n",
        "# }"
      ],
      "metadata": {
        "id": "k529uyfQvN3m"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# AdministratorAccess\n",
        "# AmazonBedrockFullAccess\n",
        "# AWSLambdaBasicExecutionRole"
      ],
      "metadata": {
        "id": "YWBz06gbvRvt"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# You have to create a dynamodb table where you will store your logs ."
      ],
      "metadata": {
        "id": "CEQnpYSOvk3K"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}