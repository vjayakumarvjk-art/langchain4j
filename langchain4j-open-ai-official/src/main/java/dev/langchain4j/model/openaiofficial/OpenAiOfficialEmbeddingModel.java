package dev.langchain4j.model.openaiofficial;

import static dev.langchain4j.internal.Utils.getOrDefault;
import static dev.langchain4j.internal.ValidationUtils.ensureGreaterThanZero;
import static dev.langchain4j.model.openaiofficial.InternalOpenAiOfficialHelper.tokenUsageFrom;
import static dev.langchain4j.model.openaiofficial.setup.OpenAiOfficialSetup.setupSyncClient;
import static java.util.stream.Collectors.toList;

import com.openai.azure.AzureOpenAIServiceVersion;
import com.openai.client.OpenAIClient;
import com.openai.credential.Credential;
import com.openai.models.embeddings.CreateEmbeddingResponse;
import com.openai.models.embeddings.EmbeddingCreateParams;
import com.openai.models.embeddings.EmbeddingModel;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.DimensionAwareEmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.output.TokenUsage;
import java.net.Proxy;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class OpenAiOfficialEmbeddingModel extends DimensionAwareEmbeddingModel {

    private final OpenAIClient client;
    private final String modelName;
    private final Integer dimensions;
    private final String user;
    private final Integer maxSegmentsPerBatch;

    public OpenAiOfficialEmbeddingModel(Builder builder) {

        if (builder.openAIClient != null) {
            this.client = builder.openAIClient;
        } else {
            this.client = setupSyncClient(
                    builder.baseUrl,
                    builder.apiKey,
                    builder.credential,
                    builder.microsoftFoundryDeploymentName,
                    builder.azureOpenAIServiceVersion,
                    builder.organizationId,
                    builder.isMicrosoftFoundry,
                    builder.isGitHubModels,
                    builder.modelName,
                    builder.timeout,
                    builder.maxRetries,
                    builder.proxy,
                    builder.customHeaders);
        }
        this.modelName = builder.modelName;
        this.dimensions = getOrDefault(builder.dimensions, knownDimension());
        this.user = builder.user;
        this.maxSegmentsPerBatch = getOrDefault(builder.maxSegmentsPerBatch, 2048);
        ensureGreaterThanZero(this.maxSegmentsPerBatch, "maxSegmentsPerBatch");
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {

        List<String> texts = textSegments.stream().map(TextSegment::text).collect(toList());

        List<List<String>> textBatches = partition(texts, maxSegmentsPerBatch);

        return embedBatchedTexts(textBatches);
    }

    @Override
    public String modelName() {
        return this.modelName;
    }

    private List<List<String>> partition(List<String> inputList, int size) {
        List<List<String>> result = new ArrayList<>();
        for (int i = 0; i < inputList.size(); i += size) {
            int toIndex = Math.min(i + size, inputList.size());
            result.add(inputList.subList(i, toIndex));
        }
        return result;
    }

    private Response<List<Embedding>> embedBatchedTexts(List<List<String>> textBatches) {
        List<Response<List<Embedding>>> responses = new ArrayList<>();
        for (List<String> batch : textBatches) {
            Response<List<Embedding>> response = embedTexts(batch);
            responses.add(response);
        }
        return Response.from(
                responses.stream()
                        .flatMap(response -> response.content().stream())
                        .toList(),
                responses.stream()
                        .map(Response::tokenUsage)
                        .filter(Objects::nonNull)
                        .reduce(TokenUsage::add)
                        .orElse(null));
    }

    private Response<List<Embedding>> embedTexts(List<String> texts) {

        EmbeddingCreateParams.Input input = EmbeddingCreateParams.Input.ofArrayOfStrings(texts);

        EmbeddingCreateParams.Builder embeddingCreateParamsBuilder = EmbeddingCreateParams.builder();
        embeddingCreateParamsBuilder.input(input);
        embeddingCreateParamsBuilder.model(modelName);
        if (user != null) {
            embeddingCreateParamsBuilder.user(user);
        }
        if (dimensions != null) {
            embeddingCreateParamsBuilder.dimensions(dimensions);
        }

        final CreateEmbeddingResponse createEmbeddingResponse =
                client.embeddings().create(embeddingCreateParamsBuilder.build());

        List<Embedding> embeddings = createEmbeddingResponse.data().stream()
                .map(embeddingItem -> Embedding.from(embeddingItem.embedding()))
                .toList();

        return Response.from(embeddings, tokenUsageFrom(createEmbeddingResponse.usage()));
    }

    public static Builder builder() {
        return new Builder();
    }

    @Override
    protected Integer knownDimension() {
        if (dimensions != null) {
            return dimensions;
        } else {
            return OpenAiOfficialEmbeddingModelName.knownDimension(modelName);
        }
    }

    public static class Builder {

        private String baseUrl;
        private String apiKey;
        private Credential credential;
        private String microsoftFoundryDeploymentName;
        private AzureOpenAIServiceVersion azureOpenAIServiceVersion;
        private String organizationId;
        private boolean isMicrosoftFoundry;
        private boolean isGitHubModels;
        private OpenAIClient openAIClient;
        private String modelName;
        private Integer dimensions;
        private String user;
        private Integer maxSegmentsPerBatch;
        private Duration timeout;
        private Integer maxRetries;
        private Proxy proxy;
        private Map<String, String> customHeaders;

        /**
         * Sets the base URL of the OpenAI-compatible API. Defaults to {@code https://api.openai.com/v1}.
         *
         * @param baseUrl the base URL
         * @return {@code this}
         */
        public Builder baseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
            return this;
        }

        /**
         * Sets the OpenAI API key used to authenticate requests.
         *
         * @param apiKey the API key
         * @return {@code this}
         */
        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        /**
         * Sets the {@link Credential} used to authenticate requests (alternative to {@link #apiKey(String)}).
         *
         * @param credential the credential
         * @return {@code this}
         */
        public Builder credential(Credential credential) {
            this.credential = credential;
            return this;
        }

        /**
         * @deprecated Use {@link #microsoftFoundryDeploymentName(String)} instead
         */
        @Deprecated
        public Builder azureDeploymentName(String azureDeploymentName) {
            this.microsoftFoundryDeploymentName = azureDeploymentName;
            return this;
        }

        /**
         * Sets the Microsoft Foundry deployment name used when connecting to Azure OpenAI or Microsoft Foundry.
         *
         * @param microsoftFoundryDeploymentName the deployment name
         * @return {@code this}
         */
        public Builder microsoftFoundryDeploymentName(String microsoftFoundryDeploymentName) {
            this.microsoftFoundryDeploymentName = microsoftFoundryDeploymentName;
            return this;
        }

        /**
         * Sets the Azure OpenAI service API version when connecting to Azure OpenAI.
         *
         * @param azureOpenAIServiceVersion the Azure OpenAI service version
         * @return {@code this}
         */
        public Builder azureOpenAIServiceVersion(AzureOpenAIServiceVersion azureOpenAIServiceVersion) {
            this.azureOpenAIServiceVersion = azureOpenAIServiceVersion;
            return this;
        }

        /**
         * Sets the OpenAI organization ID sent with each request.
         *
         * @param organizationId the organization ID
         * @return {@code this}
         */
        public Builder organizationId(String organizationId) {
            this.organizationId = organizationId;
            return this;
        }

        /**
         * @deprecated Use {@link #isMicrosoftFoundry(boolean)} instead
         */
        @Deprecated
        public Builder isAzure(boolean isAzure) {
            this.isMicrosoftFoundry = isAzure;
            return this;
        }

        /**
         * Configures the client to use Microsoft Foundry as the API provider.
         *
         * @param isMicrosoftFoundry {@code true} to use Microsoft Foundry
         * @return {@code this}
         */
        public Builder isMicrosoftFoundry(boolean isMicrosoftFoundry) {
            this.isMicrosoftFoundry = isMicrosoftFoundry;
            return this;
        }

        /**
         * Configures the client to use GitHub Models as the API provider.
         *
         * @param isGitHubModels {@code true} to use GitHub Models
         * @return {@code this}
         */
        public Builder isGitHubModels(boolean isGitHubModels) {
            this.isGitHubModels = isGitHubModels;
            return this;
        }

        /**
         * Sets a pre-configured {@link OpenAIClient} to use directly, bypassing all other connection settings.
         *
         * @param openAIClient the pre-configured client
         * @return {@code this}
         */
        public Builder openAIClient(OpenAIClient openAIClient) {
            this.openAIClient = openAIClient;
            return this;
        }

        /**
         * Sets the model name, e.g. {@code "text-embedding-3-small"}.
         *
         * @param modelName the model name
         * @return {@code this}
         */
        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        /**
         * Sets the model name using an {@link EmbeddingModel} enum constant.
         *
         * @param modelName the model name enum value
         * @return {@code this}
         */
        public Builder modelName(EmbeddingModel modelName) {
            this.modelName = modelName.toString();
            return this;
        }

        /**
         * Sets the number of dimensions for the output embedding vectors.
         * Only supported by {@code text-embedding-3} and later models.
         *
         * @param dimensions the number of output dimensions
         * @return {@code this}
         */
        public Builder dimensions(Integer dimensions) {
            this.dimensions = dimensions;
            return this;
        }

        /**
         * Sets a unique end-user identifier sent to OpenAI to help monitor and detect abuse.
         *
         * @param user the end-user identifier
         * @return {@code this}
         */
        public Builder user(String user) {
            this.user = user;
            return this;
        }

        /**
         * Sets the maximum number of text segments sent in a single embedding request. Defaults to {@code 2048}.
         *
         * @param maxSegmentsPerBatch the maximum batch size
         * @return {@code this}
         */
        public Builder maxSegmentsPerBatch(Integer maxSegmentsPerBatch) {
            this.maxSegmentsPerBatch = maxSegmentsPerBatch;
            return this;
        }

        /**
         * Sets the HTTP request timeout.
         *
         * @param timeout the request timeout
         * @return {@code this}
         */
        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        /**
         * Sets the maximum number of retries on transient errors.
         *
         * @param maxRetries the maximum number of retries
         * @return {@code this}
         */
        public Builder maxRetries(Integer maxRetries) {
            this.maxRetries = maxRetries;
            return this;
        }

        /**
         * Sets the network proxy used for HTTP connections.
         *
         * @param proxy the network proxy
         * @return {@code this}
         */
        public Builder proxy(Proxy proxy) {
            this.proxy = proxy;
            return this;
        }

        /**
         * Sets additional HTTP headers sent with every request.
         *
         * @param customHeaders the custom headers map
         * @return {@code this}
         */
        public Builder customHeaders(Map<String, String> customHeaders) {
            this.customHeaders = customHeaders;
            return this;
        }

        public OpenAiOfficialEmbeddingModel build() {
            return new OpenAiOfficialEmbeddingModel(this);
        }
    }
}
