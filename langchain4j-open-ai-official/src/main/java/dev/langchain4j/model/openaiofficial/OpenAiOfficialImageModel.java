package dev.langchain4j.model.openaiofficial;

import static dev.langchain4j.model.openaiofficial.setup.OpenAiOfficialSetup.setupSyncClient;

import com.openai.azure.AzureOpenAIServiceVersion;
import com.openai.client.OpenAIClient;
import com.openai.core.RequestOptions;
import com.openai.credential.Credential;
import com.openai.models.images.ImageGenerateParams;
import com.openai.models.images.ImagesResponse;
import dev.langchain4j.data.image.Image;
import dev.langchain4j.model.image.ImageModel;
import dev.langchain4j.model.output.Response;
import java.net.Proxy;
import java.time.Duration;
import java.util.List;
import java.util.Map;

/**
 * Represents an OpenAI image generation model.
 * Find the parameters description <a href="https://developers.openai.com/api/reference/resources/images/methods/generate">here</a>.
 */
public class OpenAiOfficialImageModel implements ImageModel {

    private final OpenAIClient client;
    private final String modelName;
    private final ImageGenerateParams.Size size;
    private final ImageGenerateParams.Quality quality;
    private final String user;
    private final Duration timeout;
    private final ImageGenerateParams.Background background;
    private final ImageGenerateParams.OutputFormat outputFormat;
    private final Long outputCompression;
    private final ImageGenerateParams.Moderation moderation;

    public OpenAiOfficialImageModel(Builder builder) {

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
        this.size = builder.size;
        this.quality = builder.quality;
        this.user = builder.user;
        this.timeout = builder.timeout;
        this.background = builder.background;
        this.outputFormat = builder.outputFormat;
        this.outputCompression = builder.outputCompression;
        this.moderation = builder.moderation;
    }

    public String modelName() {
        return modelName;
    }

    @Override
    public Response<Image> generate(String prompt) {

        ImageGenerateParams imageGenerateParams =
                imageGenerateParamsBuilder(prompt).build();

        ImagesResponse response = client.images().generate(imageGenerateParams, requestOptions());

        if (response.data().isEmpty() || response.data().get().isEmpty()) {
            throw new IllegalArgumentException("Image generation failed: no image returned");
        }

        String mimeType = response.outputFormat().map(of -> "image/" + of).orElse(null);
        return Response.from(fromOpenAiImage(response.data().get().get(0), mimeType));
    }

    @Override
    public Response<List<Image>> generate(String prompt, int n) {

        ImageGenerateParams imageGenerateParams =
                imageGenerateParamsBuilder(prompt).n(n).build();

        ImagesResponse response = client.images().generate(imageGenerateParams, requestOptions());

        if (response.data().isEmpty()) {
            throw new IllegalArgumentException("Image generation failed: no image returned");
        }

        String mimeType = response.outputFormat().map(of -> "image/" + of).orElse(null);
        return Response.from(response.data().get().stream()
                .map(img -> fromOpenAiImage(img, mimeType))
                .toList());
    }

    private ImageGenerateParams.Builder imageGenerateParamsBuilder(String prompt) {
        ImageGenerateParams.Builder builder = ImageGenerateParams.builder();
        builder.model(modelName);
        builder.prompt(prompt);

        if (size != null) {
            builder.size(size);
        }
        if (quality != null) {
            builder.quality(quality);
        }
        if (user != null) {
            builder.user(user);
        }
        if (background != null) {
            builder.background(background);
        }
        if (outputFormat != null) {
            builder.outputFormat(outputFormat);
        }
        if (outputCompression != null) {
            builder.outputCompression(outputCompression);
        }
        if (moderation != null) {
            builder.moderation(moderation);
        }
        return builder;
    }

    private RequestOptions requestOptions() {
        RequestOptions.Builder builder = new RequestOptions.Builder();
        if (timeout != null) {
            builder.timeout(timeout);
        }
        return builder.build();
    }

    private static Image fromOpenAiImage(com.openai.models.images.Image openAiImage, String mimeType) {
        Image.Builder imageBuilder = Image.builder();

        if (openAiImage.url().isPresent()) {
            imageBuilder.url(openAiImage.url().get());
        } else if (openAiImage.b64Json().isPresent()) {
            imageBuilder.base64Data(openAiImage.b64Json().get());
        } else {
            throw new IllegalArgumentException("Image must have either a URL or base64 data");
        }

        if (openAiImage.revisedPrompt().isPresent()) {
            imageBuilder.revisedPrompt(openAiImage.revisedPrompt().get());
        }

        if (mimeType != null) {
            imageBuilder.mimeType(mimeType);
        }

        return imageBuilder.build();
    }

    public static Builder builder() {
        return new Builder();
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
        private ImageGenerateParams.Size size;
        private ImageGenerateParams.Quality quality;
        private String user;
        private Duration timeout;
        private Integer maxRetries;
        private Proxy proxy;
        private Map<String, String> customHeaders;
        private ImageGenerateParams.Background background;
        private ImageGenerateParams.OutputFormat outputFormat;
        private Long outputCompression;
        private ImageGenerateParams.Moderation moderation;

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
         * Sets the model name, e.g. {@code "dall-e-3"} or {@code "gpt-image-1"}.
         *
         * @param modelName the model name
         * @return {@code this}
         */
        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        /**
         * Sets the model name using a {@link com.openai.models.images.ImageModel} enum constant.
         *
         * @param modelName the model name enum value
         * @return {@code this}
         */
        public Builder modelName(com.openai.models.images.ImageModel modelName) {
            this.modelName = modelName.toString();
            return this;
        }

        /**
         * Sets the image size as a string, e.g. {@code "1024x1024"} or {@code "1792x1024"}.
         *
         * @param size the image size string
         * @return {@code this}
         */
        public Builder size(String size) {
            this.size = ImageGenerateParams.Size.of(size);
            return this;
        }

        /**
         * Sets the image size using an {@link ImageGenerateParams.Size} enum constant.
         *
         * @param size the image size
         * @return {@code this}
         */
        public Builder size(ImageGenerateParams.Size size) {
            this.size = size;
            return this;
        }

        /**
         * Sets the image quality as a string, e.g. {@code "standard"} or {@code "hd"}.
         *
         * @param quality the image quality string
         * @return {@code this}
         */
        public Builder quality(String quality) {
            this.quality = ImageGenerateParams.Quality.of(quality);
            return this;
        }

        /**
         * Sets the image quality using an {@link ImageGenerateParams.Quality} enum constant.
         *
         * @param quality the image quality
         * @return {@code this}
         */
        public Builder quality(ImageGenerateParams.Quality quality) {
            this.quality = quality;
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
         * Sets the background type as a string, e.g. {@code "transparent"} or {@code "opaque"}.
         *
         * @param background the background type string
         * @return {@code this}
         */
        public Builder background(String background) {
            this.background = ImageGenerateParams.Background.of(background);
            return this;
        }

        /**
         * Sets the background type using an {@link ImageGenerateParams.Background} enum constant.
         *
         * @param background the background type
         * @return {@code this}
         */
        public Builder background(ImageGenerateParams.Background background) {
            this.background = background;
            return this;
        }

        /**
         * Sets the output format as a string, e.g. {@code "png"}, {@code "webp"}, or {@code "jpeg"}.
         *
         * @param outputFormat the output format string
         * @return {@code this}
         */
        public Builder outputFormat(String outputFormat) {
            this.outputFormat = ImageGenerateParams.OutputFormat.of(outputFormat);
            return this;
        }

        /**
         * Sets the output format using an {@link ImageGenerateParams.OutputFormat} enum constant.
         *
         * @param outputFormat the output format
         * @return {@code this}
         */
        public Builder outputFormat(ImageGenerateParams.OutputFormat outputFormat) {
            this.outputFormat = outputFormat;
            return this;
        }

        /**
         * Sets the compression level for the output image (0–100).
         * Only applicable to {@code webp} and {@code jpeg} output formats.
         *
         * @param outputCompression the compression level
         * @return {@code this}
         */
        public Builder outputCompression(Long outputCompression) {
            this.outputCompression = outputCompression;
            return this;
        }

        /**
         * Sets the content moderation level as a string, e.g. {@code "low"} or {@code "auto"}.
         *
         * @param moderation the moderation level string
         * @return {@code this}
         */
        public Builder moderation(String moderation) {
            this.moderation = ImageGenerateParams.Moderation.of(moderation);
            return this;
        }

        /**
         * Sets the content moderation level using an {@link ImageGenerateParams.Moderation} enum constant.
         *
         * @param moderation the moderation level
         * @return {@code this}
         */
        public Builder moderation(ImageGenerateParams.Moderation moderation) {
            this.moderation = moderation;
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

        public OpenAiOfficialImageModel build() {
            return new OpenAiOfficialImageModel(this);
        }
    }
}
