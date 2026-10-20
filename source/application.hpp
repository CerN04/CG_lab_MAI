#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "graphics_internal.hpp"

#include <vector>
#include <glm/glm.hpp>

namespace application {

struct Vertex {
	glm::vec3 pos;
	glm::vec3 normal;
};

struct PointLight {
	glm::vec4 position;
	glm::vec4 color;
};

struct SpotLight {
	glm::vec4 position;
	glm::vec4 direction;
	glm::vec4 color;
	glm::vec4 angles;
};

struct SceneUBO {
	alignas(16) glm::mat4 viewProj;
	alignas(16) glm::vec3 cameraPos;
	alignas(4)  float ambientStrength;
	alignas(16) glm::vec3 ambientColor;
	alignas(4)  float _pad1;
	alignas(16) glm::vec3 dirLightDirection;
	alignas(4)  float _pad2;
	alignas(16) glm::vec3 dirLightColor;
	alignas(4)  float dirLightIntensity;
	alignas(4)  int numPointLights;
	alignas(4)  int numSpotLights;
	alignas(4)  int showLightGizmos;
	alignas(4)  float _pad3;
};

struct ObjectUBO {
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 normalMatrix;
	alignas(16) glm::vec4 albedo;
	alignas(16) glm::vec4 specularColor;
	alignas(4)  float shininess;
	alignas(4)  int useLighting;
	alignas(4)  float _pad0;
	alignas(4)  float _pad1;
};

struct PointLightState {
	glm::vec3 position{0.0f, 3.0f, 0.0f};
	glm::vec3 color{1.0f, 1.0f, 1.0f};
	float intensity{1.0f};
};

struct SpotLightState {
	glm::vec3 position{0.0f, 5.0f, 0.0f};
	glm::vec3 direction{0.0f, -1.0f, 0.0f};
	glm::vec3 color{1.0f, 1.0f, 1.0f};
	float intensity{1.0f};
	float innerAngleDeg{20.0f};
	float outerAngleDeg{30.0f};
};

struct ObjectState {
	glm::vec3 position{0.0f, 0.0f, 0.0f};
	glm::vec3 rotationDeg{0.0f, 0.0f, 0.0f};
	glm::vec3 scale{1.0f, 1.0f, 1.0f};
	glm::vec3 albedo{1.0f, 1.0f, 1.0f};
	glm::vec3 specularColor{1.0f, 1.0f, 1.0f};
	float shininess{32.0f};
	bool useLighting{true};
	bool animPlaying = false;
	float animSpeed = 1.0f;
	float animRadius = 2.0f;
	double animTime = 0.0;

	VkBuffer uniformBuffer = VK_NULL_HANDLE;
	VmaAllocation uniformAllocation = nullptr;
	void* uniformMapped = nullptr;
	VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
	ObjectUBO ubo{};
};

struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	VkBuffer vertexBuffer = VK_NULL_HANDLE;
	VmaAllocation vertexAllocation = nullptr;
	VkBuffer indexBuffer = VK_NULL_HANDLE;
	VmaAllocation indexAllocation = nullptr;
};

struct CameraState {
	glm::vec3 position{0.0f, 5.0f, 15.0f};
	float yaw{-90.0f};
	float pitch{-15.0f};
	float fov{45.0f};
	float nearPlane{0.1f};
	float farPlane{100.0f};
};

struct Resources {
	Mesh boxMesh;
	Mesh floorMesh;
	Mesh postMesh;
	Mesh coneMesh;

	VkBuffer sceneUniformBuffer = VK_NULL_HANDLE;
	VmaAllocation sceneUniformAllocation = nullptr;
	void* sceneUniformMapped = nullptr;
	SceneUBO sceneUbo{};

	VkBuffer pointLightSSBO = VK_NULL_HANDLE;
	VmaAllocation pointLightSSBOAllocation = nullptr;
	void* pointLightSSBOMapped = nullptr;
	VkDeviceSize pointLightSSBOSize = 0;

	VkBuffer spotLightSSBO = VK_NULL_HANDLE;
	VmaAllocation spotLightSSBOAllocation = nullptr;
	void* spotLightSSBOMapped = nullptr;
	VkDeviceSize spotLightSSBOSize = 0;

	VkDescriptorSetLayout sceneSetLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout objectSetLayout = VK_NULL_HANDLE;
	VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
	VkDescriptorSet sceneDescriptorSet = VK_NULL_HANDLE;

	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkPipeline pipeline = VK_NULL_HANDLE;

	CameraState camera;

	glm::vec3 ambientColor{1.0f, 1.0f, 1.0f};
	float ambientStrength{0.1f};

	glm::vec3 dirLightDirection{-0.3f, -1.0f, -0.5f};
	glm::vec3 dirLightColor{1.0f, 1.0f, 1.0f};
	float dirLightIntensity{0.8f};

	std::vector<PointLightState> pointLights;
	std::vector<SpotLightState> spotLights;
	bool showLightGizmos{true};

	std::vector<ObjectState> objects;
	std::vector<ObjectState> gizmoObjects;

	double lastTime = -1.0;
};

extern Resources res;

bool initialize();
void shutdown();

void update(double time);
void render(const graphics::internal::FrameData& fd);

} // namespace application