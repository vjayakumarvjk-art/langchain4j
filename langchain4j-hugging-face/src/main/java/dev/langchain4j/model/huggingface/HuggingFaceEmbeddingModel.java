package dev.langchain4j.model.huggingface;

import static dev.langchain4j.internal.ValidationUtils.ensureNotBlank;
import static dev.langchain4j.spi.ServiceHelper.loadFactories;
import static java.util.stream.Collectors.toList;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.DimensionAwareEmbeddingModel;
import dev.langchain4j.model.huggingface.client.EmbeddingRequest;
import dev.langchain4j.model.huggingface.client.HuggingFaceClient;
import dev.langchain4j.model.huggingface.spi.HuggingFaceClientFactory;
import dev.langchain4j.model.huggingface.spi.HuggingFaceEmbeddingModelBuilderFactory;
import dev.langchain4j.model.output.Response;
import java.time.Duration;
import java.util.List;

public class HuggingFaceEmbeddingModel extends DimensionAwareEmbeddingModel {

    private static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(15);

    private HuggingFaceClient client;
    private final boolean waitForModel;
    private final String modelId;
    private final String baseUrl;

    /**
     * Constructor with Custom baseUrl parameter
     */
    public HuggingFaceEmbeddingModel(
            String baseUrl, String accessToken, String modelId, Boolean waitForModel, Duration timeout) {
        ensureNotBlank(
                accessToken,
                "%s",
                "HuggingFace access token must be defined. It can be generated here: https://huggingface.co/settings/tokens");
        this.waitForModel = waitForModel == null || waitForModel;
        this.baseUrl = baseUrl;
        this.modelId = modelId;
        this.client = createClient(accessToken, modelId, timeout);
    }

    public HuggingFaceEmbeddingModel(String accessToken, String modelId, Boolean waitForModel, Duration timeout) {
        this(null, accessToken, modelId, waitForModel, timeout);
    }

    private HuggingFaceClient createClient(String accessToken, String modelId, Duration timeout) {
        return FactoryCreator.FACTORY.create(new HuggingFaceClientFactory.Input() {
            @Override
            public String baseUrl() {
                return baseUrl;
            }

            @Override
            public String apiKey() {
                return accessToken;
            }

            @Override
            public String modelId() {
                return modelId;
            }

            @Override
            public Duration timeout() {
                return timeout == null ? DEFAULT_TIMEOUT : timeout;
            }
        });
    }

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> textSegments) {

        List<String> texts = textSegments.stream().map(TextSegment::text).collect(toList());

        return embedTexts(texts);
    }

    private Response<List<Embedding>> embedTexts(List<String> texts) {

        EmbeddingRequest request = new EmbeddingRequest(texts, waitForModel);

        List<float[]> response = client.embed(request);

        List<Embedding> embeddings = response.stream().map(Embedding::from).collect(toList());

        return Response.from(embeddings);
    }

    public static HuggingFaceEmbeddingModel withAccessToken(String accessToken) {
        return builder().accessToken(accessToken).build();
    }

    public static HuggingFaceEmbeddingModelBuilder builder() {
        for (HuggingFaceEmbeddingModelBuilderFactory factory :
                loadFactories(HuggingFaceEmbeddingModelBuilderFactory.class)) {
            return factory.get();
        }
        return new HuggingFaceEmbeddingModelBuilder();
    }

    public static class HuggingFaceEmbeddingModelBuilder {
        private String baseUrl;
        private String accessToken;
        private String modelId;
        private Boolean waitForModel;
        private Duration timeout;

        public HuggingFaceEmbeddingModelBuilder() {
            // This is public so it can be extended
        }

        /**
         * Sets the base URL of the Hugging Face Inference API.
         * Defaults to the standard Hugging Face API endpoint.
         *
         * @param baseUrl the base URL
         * @return {@code this}
         */
        public HuggingFaceEmbeddingModelBuilder baseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
            return this;
        }

        /**
         * Sets the Hugging Face access token used to authenticate requests.
         * Tokens can be generated at <a href="https://huggingface.co/settings/tokens">https://huggingface.co/settings/tokens</a>.
         *
         * @param accessToken the Hugging Face access token
         * @return {@code this}
         */
        public HuggingFaceEmbeddingModelBuilder accessToken(String accessToken) {
            this.accessToken = accessToken;
            return this;
        }

        /**
         * Sets the Hugging Face model ID to use for embeddings,
         * e.g. {@code "sentence-transformers/all-MiniLM-L6-v2"}.
         *
         * @param modelId the model ID
         * @return {@code this}
         */
        public HuggingFaceEmbeddingModelBuilder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        /**
         * Sets whether to wait for the model to load if it is not ready yet.
         * Defaults to {@code true}.
         *
         * @param waitForModel {@code true} to wait for the model to be ready
         * @return {@code this}
         */
        public HuggingFaceEmbeddingModelBuilder waitForModel(Boolean waitForModel) {
            this.waitForModel = waitForModel;
            return this;
        }

        /**
         * Sets the HTTP request timeout. Defaults to 15 seconds.
         *
         * @param timeout the request timeout
         * @return {@code this}
         */
        public HuggingFaceEmbeddingModelBuilder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        public HuggingFaceEmbeddingModel build() {
            return new HuggingFaceEmbeddingModel(
                    this.baseUrl, this.accessToken, this.modelId, this.waitForModel, this.timeout);
        }

        public String toString() {
            return "HuggingFaceEmbeddingModel.HuggingFaceEmbeddingModelBuilder(baseUrl=" + this.baseUrl
                    + ", accessToken=" + this.accessToken + ", modelId=" + this.modelId + ", waitForModel="
                    + this.waitForModel + ", timeout=" + this.timeout + ")";
        }
    }
}
