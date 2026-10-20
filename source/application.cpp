#include "application.hpp"
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <imgui.h>
#include <glm/gtc/matrix_transform.hpp>

namespace application {

Resources res;

namespace {

constexpr VkDeviceSize MIN_SSBO_SIZE = 256;

void generateBoxWithNormals(Mesh& mesh, float width, float height, float depth) {
    const float w = width * 0.5f;
    const float h = height * 0.5f;
    const float d = depth * 0.5f;

    mesh.vertices.clear();
    mesh.indices.clear();

    struct Face { glm::vec3 normal; glm::vec3 corners[4]; };
    Face faces[6] = {
        { {0,0,-1}, { {-w,-h,-d}, {-w,+h,-d}, {+w,+h,-d}, {+w,-h,-d} } },
        { {0,0,+1}, { {-w,-h,+d}, {+w,-h,+d}, {+w,+h,+d}, {-w,+h,+d} } },
        { {-1,0,0}, { {-w,-h,-d}, {-w,-h,+d}, {-w,+h,+d}, {-w,+h,-d} } },
        { {+1,0,0}, { {+w,-h,-d}, {+w,+h,-d}, {+w,+h,+d}, {+w,-h,+d} } },
        { {0,-1,0}, { {-w,-h,-d}, {+w,-h,-d}, {+w,-h,+d}, {-w,-h,+d} } },
        { {0,+1,0}, { {-w,+h,-d}, {-w,+h,+d}, {+w,+h,+d}, {+w,+h,-d} } },
    };

    for (int f = 0; f < 6; ++f) {
        uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
        for (int c = 0; c < 4; ++c) {
            Vertex v{};
            v.pos = faces[f].corners[c];
            v.normal = faces[f].normal;
            mesh.vertices.push_back(v);
        }
        mesh.indices.push_back(base + 0);
        mesh.indices.push_back(base + 1);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 2);
        mesh.indices.push_back(base + 3);
        mesh.indices.push_back(base + 0);
    }
}

void generatePlane(Mesh& mesh, float sizeX, float sizeZ) {
    mesh.vertices.clear();
    mesh.indices.clear();

    float hx = sizeX * 0.5f;
    float hz = sizeZ * 0.5f;
    glm::vec3 n = {0, 1, 0};

    mesh.vertices = {
        {{-hx, 0, -hz}, n},
        {{+hx, 0, -hz}, n},
        {{+hx, 0, +hz}, n},
        {{-hx, 0, +hz}, n},
    };
    mesh.indices = {0, 2, 1, 0, 3, 2};
}

void generateCone(Mesh& mesh, float radius, float height, int segments) {
    mesh.vertices.clear();
    mesh.indices.clear();

    glm::vec3 tip = {0, 0, 0};

    for (int i = 0; i < segments; ++i) {
        float a0 = glm::radians(360.0f * float(i) / float(segments));
        float a1 = glm::radians(360.0f * float(i + 1) / float(segments));

        glm::vec3 p0 = {radius * cosf(a0), -height, radius * sinf(a0)};
        glm::vec3 p1 = {radius * cosf(a1), -height, radius * sinf(a1)};

        glm::vec3 edge0 = p0 - tip;
        glm::vec3 edge1 = p1 - tip;
        glm::vec3 n = glm::normalize(glm::cross(edge0, edge1));

        uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back({tip, n});
        mesh.vertices.push_back({p0, n});
        mesh.vertices.push_back({p1, n});
        mesh.indices.push_back(base);
        mesh.indices.push_back(base + 1);
        mesh.indices.push_back(base + 2);
    }
}

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                  VmaAllocationCreateFlags allocFlags,
                  VkBuffer& buffer, VmaAllocation& allocation,
                  void** mapped = nullptr) {
    auto& ctx = graphics::internal::context;

    const VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    const VmaAllocationCreateInfo allocInfo = {
        .flags = allocFlags,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    VmaAllocationInfo allocResult{};
    if (vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocInfo,
                        &buffer, &allocation, &allocResult) != VK_SUCCESS) {
        std::cerr << "Failed to create buffer\n";
        return false;
    }

    if (mapped != nullptr) {
        *mapped = allocResult.pMappedData;
    }
    return true;
}

VkShaderModule createShaderModule(const char* path) {
    auto& ctx = graphics::internal::context;

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader: " << path << "\n";
        return VK_NULL_HANDLE;
    }

    const auto size = static_cast<size_t>(file.tellg());
    std::vector<char> code(size);
    file.seekg(0);
    file.read(code.data(), static_cast<std::streamsize>(size));

    const VkShaderModuleCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const uint32_t*>(code.data()),
    };

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(ctx.device, &createInfo, nullptr, &module) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module: " << path << "\n";
        return VK_NULL_HANDLE;
    }
    return module;
}

bool uploadMesh(Mesh& mesh) {
    auto& ctx = graphics::internal::context;

    if (!createBuffer(mesh.vertices.size() * sizeof(Vertex),
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                      mesh.vertexBuffer, mesh.vertexAllocation)) return false;

    if (!createBuffer(mesh.indices.size() * sizeof(uint32_t),
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                      mesh.indexBuffer, mesh.indexAllocation)) return false;

    void* data = nullptr;
    vmaMapMemory(ctx.allocator, mesh.vertexAllocation, &data);
    std::memcpy(data, mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex));
    vmaUnmapMemory(ctx.allocator, mesh.vertexAllocation);

    vmaMapMemory(ctx.allocator, mesh.indexAllocation, &data);
    std::memcpy(data, mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t));
    vmaUnmapMemory(ctx.allocator, mesh.indexAllocation);

    return true;
}

void destroyMesh(Mesh& mesh) {
    auto& ctx = graphics::internal::context;
    vmaDestroyBuffer(ctx.allocator, mesh.vertexBuffer, mesh.vertexAllocation);
    vmaDestroyBuffer(ctx.allocator, mesh.indexBuffer, mesh.indexAllocation);
}

bool createObject(ObjectState& obj, const glm::vec3& initialPos, const glm::vec3& albedo) {
    auto& ctx = graphics::internal::context;

    obj.position = initialPos;
    obj.albedo = albedo;

    if (!createBuffer(sizeof(ObjectUBO),
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VMA_ALLOCATION_CREATE_MAPPED_BIT |
                          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                      obj.uniformBuffer, obj.uniformAllocation, &obj.uniformMapped)) return false;

    VkDescriptorSetAllocateInfo setAllocInfo{};
    setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAllocInfo.descriptorPool = res.descriptorPool;
    setAllocInfo.descriptorSetCount = 1;
    setAllocInfo.pSetLayouts = &res.objectSetLayout;

    if (vkAllocateDescriptorSets(ctx.device, &setAllocInfo, &obj.descriptorSet) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor set for object\n";
        return false;
    }

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = obj.uniformBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(ObjectUBO);

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = obj.descriptorSet;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    return true;
}

void destroyObject(ObjectState& obj) {
    auto& ctx = graphics::internal::context;
    if (obj.descriptorSet != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(ctx.device, res.descriptorPool, 1, &obj.descriptorSet);
        obj.descriptorSet = VK_NULL_HANDLE;
    }
    vmaDestroyBuffer(ctx.allocator, obj.uniformBuffer, obj.uniformAllocation);
}

void rebuildSSBOs() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);

    if (res.pointLightSSBO != VK_NULL_HANDLE) {
        vmaDestroyBuffer(ctx.allocator, res.pointLightSSBO, res.pointLightSSBOAllocation);
        res.pointLightSSBO = VK_NULL_HANDLE;
    }

    VkDeviceSize pointSize = std::max(MIN_SSBO_SIZE, static_cast<VkDeviceSize>(
        std::max(size_t(1), res.pointLights.size()) * sizeof(PointLight)));
    createBuffer(pointSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VMA_ALLOCATION_CREATE_MAPPED_BIT |
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                 res.pointLightSSBO, res.pointLightSSBOAllocation, &res.pointLightSSBOMapped);
    res.pointLightSSBOSize = pointSize;

    if (res.spotLightSSBO != VK_NULL_HANDLE) {
        vmaDestroyBuffer(ctx.allocator, res.spotLightSSBO, res.spotLightSSBOAllocation);
        res.spotLightSSBO = VK_NULL_HANDLE;
    }

    VkDeviceSize spotSize = std::max(MIN_SSBO_SIZE, static_cast<VkDeviceSize>(
        std::max(size_t(1), res.spotLights.size()) * sizeof(SpotLight)));
    createBuffer(spotSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VMA_ALLOCATION_CREATE_MAPPED_BIT |
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                 res.spotLightSSBO, res.spotLightSSBOAllocation, &res.spotLightSSBOMapped);
    res.spotLightSSBOSize = spotSize;

    VkDescriptorBufferInfo sceneBufInfo{};
    sceneBufInfo.buffer = res.sceneUniformBuffer;
    sceneBufInfo.offset = 0;
    sceneBufInfo.range = sizeof(SceneUBO);

    VkDescriptorBufferInfo pointBufInfo{};
    pointBufInfo.buffer = res.pointLightSSBO;
    pointBufInfo.offset = 0;
    pointBufInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo spotBufInfo{};
    spotBufInfo.buffer = res.spotLightSSBO;
    spotBufInfo.offset = 0;
    spotBufInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writes[3] = {};

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = res.sceneDescriptorSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo = &sceneBufInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = res.sceneDescriptorSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &pointBufInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = res.sceneDescriptorSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &spotBufInfo;

    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, nullptr);
}

void drawMesh(VkCommandBuffer cmd, const Mesh& mesh) {
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer, &offset);
    vkCmdBindIndexBuffer(cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, uint32_t(mesh.indices.size()), 1, 0, 0, 0);
}

} // namespace

bool initialize() {
    auto& ctx = graphics::internal::context;

    generateBoxWithNormals(res.boxMesh, 2.0f, 1.0f, 3.0f);
    if (!uploadMesh(res.boxMesh)) return false;

    generatePlane(res.floorMesh, 30.0f, 30.0f);
    if (!uploadMesh(res.floorMesh)) return false;

    generateBoxWithNormals(res.postMesh, 0.15f, 5.0f, 0.15f);
    if (!uploadMesh(res.postMesh)) return false;

    generateCone(res.coneMesh, 0.3f, 0.6f, 16);
    if (!uploadMesh(res.coneMesh)) return false;

    if (!createBuffer(sizeof(SceneUBO),
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VMA_ALLOCATION_CREATE_MAPPED_BIT |
                          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                      res.sceneUniformBuffer, res.sceneUniformAllocation,
                      &res.sceneUniformMapped)) return false;

    VkDescriptorSetLayoutBinding sceneBindings[3] = {};
    sceneBindings[0].binding = 0;
    sceneBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sceneBindings[0].descriptorCount = 1;
    sceneBindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    sceneBindings[1].binding = 1;
    sceneBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sceneBindings[1].descriptorCount = 1;
    sceneBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    sceneBindings[2].binding = 2;
    sceneBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sceneBindings[2].descriptorCount = 1;
    sceneBindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo sceneLayoutInfo{};
    sceneLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    sceneLayoutInfo.bindingCount = 3;
    sceneLayoutInfo.pBindings = sceneBindings;

    vkCreateDescriptorSetLayout(ctx.device, &sceneLayoutInfo, nullptr, &res.sceneSetLayout);

    VkDescriptorSetLayoutBinding objectBinding{};
    objectBinding.binding = 0;
    objectBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    objectBinding.descriptorCount = 1;
    objectBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo objectLayoutInfo{};
    objectLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    objectLayoutInfo.bindingCount = 1;
    objectLayoutInfo.pBindings = &objectBinding;

    vkCreateDescriptorSetLayout(ctx.device, &objectLayoutInfo, nullptr, &res.objectSetLayout);

    VkDescriptorPoolSize poolSizes[2] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 50;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 10;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 60;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;

    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &res.descriptorPool);

    VkDescriptorSetAllocateInfo sceneAllocInfo{};
    sceneAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    sceneAllocInfo.descriptorPool = res.descriptorPool;
    sceneAllocInfo.descriptorSetCount = 1;
    sceneAllocInfo.pSetLayouts = &res.sceneSetLayout;

    vkAllocateDescriptorSets(ctx.device, &sceneAllocInfo, &res.sceneDescriptorSet);

    res.pointLights.push_back({{3.0f, 4.0f, 3.0f}, {1.0f, 0.8f, 0.6f}, 5.0f});

    res.spotLights.push_back({{-5.0f, 4.9f, 0.0f}, {0.0f, -1.0f, 0.2f},
                               {1.0f, 1.0f, 0.9f}, 8.0f, 15.0f, 25.0f});

    rebuildSSBOs();

    res.objects.resize(3);
    if (!createObject(res.objects[0], glm::vec3(-3.0f, 0.5f, 0.0f), glm::vec3(0.9f, 0.3f, 0.3f))) return false;
    if (!createObject(res.objects[1], glm::vec3(+3.0f, 0.5f, 0.0f), glm::vec3(0.3f, 0.7f, 1.0f))) return false;
    res.objects[1].animPlaying = true;
    res.objects[1].rotationDeg = glm::vec3(0.0f, 45.0f, 0.0f);

    if (!createObject(res.objects[2], glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.4f, 0.4f, 0.4f))) return false;
    res.objects[2].specularColor = glm::vec3(0.2f);
    res.objects[2].shininess = 8.0f;

    ObjectState postObj;
    if (!createObject(postObj, glm::vec3(-5.0f, 2.5f, 0.0f), glm::vec3(0.3f, 0.3f, 0.3f))) return false;
    postObj.specularColor = glm::vec3(0.1f);
    postObj.shininess = 4.0f;
    res.objects.push_back(postObj);

    VkDescriptorSetLayout setLayouts[2] = {res.sceneSetLayout, res.objectSetLayout};

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 2;
    layoutInfo.pSetLayouts = setLayouts;

    vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &res.pipelineLayout);

    VkShaderModule vertModule = createShaderModule("shaders/shader.vert.spv");
    VkShaderModule fragModule = createShaderModule("shaders/shader.frag.spv");
    if (vertModule == VK_NULL_HANDLE || fragModule == VK_NULL_HANDLE) return false;

    const VkPipelineShaderStageCreateInfo stages[2] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertModule, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragModule, .pName = "main" },
    };

    const VkVertexInputBindingDescription binding = {
        .binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    const VkVertexInputAttributeDescription attributes[2] = {
        { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, pos) },
        { .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal) },
    };
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 2;
    vertexInput.pVertexAttributeDescriptions = attributes;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    const VkDynamicState dynamicStates[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable = VK_FALSE;
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blendAttachment;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = res.pipelineLayout;
    pipelineInfo.renderPass = ctx.render_pass;
    pipelineInfo.subpass = 0;

    vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &res.pipeline);

    vkDestroyShaderModule(ctx.device, vertModule, nullptr);
    vkDestroyShaderModule(ctx.device, fragModule, nullptr);

    return true;
}

void update(double time) {
    auto& ctx = graphics::internal::context;

    if (res.lastTime < 0.0) res.lastTime = time;
    const double dt = time - res.lastTime;
    res.lastTime = time;

    ImGui::Begin("Camera");
    ImGui::DragFloat3("Position##cam", &res.camera.position.x, 0.1f);
    ImGui::SliderFloat("Yaw", &res.camera.yaw, -180.0f, 180.0f);
    ImGui::SliderFloat("Pitch", &res.camera.pitch, -89.0f, 89.0f);
    ImGui::SliderFloat("FOV", &res.camera.fov, 10.0f, 120.0f);
    ImGui::End();

    ImGui::Begin("Lighting");

    if (ImGui::CollapsingHeader("Ambient")) {
        ImGui::ColorEdit3("Ambient Color", &res.ambientColor.x);
        ImGui::SliderFloat("Ambient Strength", &res.ambientStrength, 0.0f, 1.0f);
    }

    if (ImGui::CollapsingHeader("Directional Light")) {
        ImGui::DragFloat3("Direction##dir", &res.dirLightDirection.x, 0.01f, -1.0f, 1.0f);
        ImGui::ColorEdit3("Color##dir", &res.dirLightColor.x);
        ImGui::SliderFloat("Intensity##dir", &res.dirLightIntensity, 0.0f, 5.0f);
    }

    if (ImGui::CollapsingHeader("Point Lights")) {
        if (ImGui::Button("Add Point Light")) {
            res.pointLights.push_back({{0.0f, 3.0f, 0.0f}, {1.0f, 1.0f, 1.0f}, 3.0f});
            rebuildSSBOs();
        }
        int removeIdx = -1;
        for (int i = 0; i < int(res.pointLights.size()); ++i) {
            ImGui::PushID(i);
            ImGui::Text("Point Light %d", i);
            ImGui::DragFloat3("Position##pl", &res.pointLights[i].position.x, 0.1f);
            ImGui::ColorEdit3("Color##pl", &res.pointLights[i].color.x);
            ImGui::SliderFloat("Intensity##pl", &res.pointLights[i].intensity, 0.0f, 50.0f);
            if (ImGui::Button("Remove##pl")) removeIdx = i;
            ImGui::Separator();
            ImGui::PopID();
        }
        if (removeIdx >= 0) {
            res.pointLights.erase(res.pointLights.begin() + removeIdx);
            rebuildSSBOs();
        }
    }

    if (ImGui::CollapsingHeader("Spot Lights")) {
        if (ImGui::Button("Add Spot Light")) {
            res.spotLights.push_back({{0.0f, 5.0f, 0.0f}, {0.0f, -1.0f, 0.0f},
                                       {1.0f, 1.0f, 1.0f}, 5.0f, 15.0f, 25.0f});
            rebuildSSBOs();
        }
        int removeIdx = -1;
        for (int i = 0; i < int(res.spotLights.size()); ++i) {
            ImGui::PushID(1000 + i);
            ImGui::Text("Spot Light %d", i);
            ImGui::DragFloat3("Position##sl", &res.spotLights[i].position.x, 0.1f);
            ImGui::DragFloat3("Direction##sl", &res.spotLights[i].direction.x, 0.01f, -1.0f, 1.0f);
            ImGui::ColorEdit3("Color##sl", &res.spotLights[i].color.x);
            ImGui::SliderFloat("Intensity##sl", &res.spotLights[i].intensity, 0.0f, 50.0f);
            ImGui::SliderFloat("Inner Angle##sl", &res.spotLights[i].innerAngleDeg, 1.0f, 89.0f);
            ImGui::SliderFloat("Outer Angle##sl", &res.spotLights[i].outerAngleDeg, 1.0f, 90.0f);
            if (res.spotLights[i].innerAngleDeg > res.spotLights[i].outerAngleDeg)
                res.spotLights[i].outerAngleDeg = res.spotLights[i].innerAngleDeg;
            if (ImGui::Button("Remove##sl")) removeIdx = i;
            ImGui::Separator();
            ImGui::PopID();
        }
        if (removeIdx >= 0) {
            res.spotLights.erase(res.spotLights.begin() + removeIdx);
            rebuildSSBOs();
        }
    }

    ImGui::Checkbox("Show Light Gizmos", &res.showLightGizmos);

    ImGui::End();

    ImGui::Begin("Objects");

    const char* objNames[] = {"Box 1", "Box 2", "Floor", "Post"};
    for (int i = 0; i < int(res.objects.size()); ++i) {
        ImGui::PushID(2000 + i);
        const char* name = (i < 4) ? objNames[i] : "Object";
        if (ImGui::CollapsingHeader(name)) {
            if (i < 2) {
                ImGui::DragFloat3("Position", &res.objects[i].position.x, 0.1f);
                ImGui::SliderFloat3("Rotation", &res.objects[i].rotationDeg.x, -180.0f, 180.0f);
                ImGui::SliderFloat3("Scale", &res.objects[i].scale.x, 0.1f, 5.0f);
            }
            ImGui::ColorEdit3("Albedo", &res.objects[i].albedo.x);
            ImGui::ColorEdit3("Specular Color", &res.objects[i].specularColor.x);
            ImGui::SliderFloat("Shininess", &res.objects[i].shininess, 1.0f, 256.0f);
            if (i == 1) {
                ImGui::Checkbox("Animate", &res.objects[i].animPlaying);
                ImGui::SliderFloat("Anim Speed", &res.objects[i].animSpeed, 0.0f, 5.0f);
                ImGui::SliderFloat("Anim Radius", &res.objects[i].animRadius, 0.0f, 10.0f);
            }
        }
        ImGui::PopID();
    }

    ImGui::End();

    glm::vec3 front;
    front.x = cosf(glm::radians(res.camera.yaw)) * cosf(glm::radians(res.camera.pitch));
    front.y = sinf(glm::radians(res.camera.pitch));
    front.z = sinf(glm::radians(res.camera.yaw)) * cosf(glm::radians(res.camera.pitch));
    front = glm::normalize(front);

    const glm::mat4 view = glm::lookAt(res.camera.position,
                                        res.camera.position + front,
                                        glm::vec3(0.0f, 1.0f, 0.0f));

    const float aspect = ctx.swapchain_extent.height > 0
        ? float(ctx.swapchain_extent.width) / float(ctx.swapchain_extent.height) : 1.0f;

    glm::mat4 proj = glm::perspective(glm::radians(res.camera.fov), aspect,
                                       res.camera.nearPlane, res.camera.farPlane);
    proj[1][1] *= -1.0f;

    res.sceneUbo.viewProj = proj * view;
    res.sceneUbo.cameraPos = res.camera.position;
    res.sceneUbo.ambientStrength = res.ambientStrength;
    res.sceneUbo.ambientColor = res.ambientColor;
    res.sceneUbo.dirLightDirection = glm::normalize(res.dirLightDirection);
    res.sceneUbo.dirLightColor = res.dirLightColor;
    res.sceneUbo.dirLightIntensity = res.dirLightIntensity;
    res.sceneUbo.numPointLights = int(res.pointLights.size());
    res.sceneUbo.numSpotLights = int(res.spotLights.size());
    res.sceneUbo.showLightGizmos = res.showLightGizmos ? 1 : 0;

    std::vector<PointLight> plGPU(res.pointLights.size());
    for (size_t i = 0; i < res.pointLights.size(); ++i) {
        plGPU[i].position = glm::vec4(res.pointLights[i].position, 0.0f);
        plGPU[i].color = glm::vec4(res.pointLights[i].color, res.pointLights[i].intensity);
    }
    if (!plGPU.empty()) {
        std::memcpy(res.pointLightSSBOMapped, plGPU.data(), plGPU.size() * sizeof(PointLight));
    }

    std::vector<SpotLight> slGPU(res.spotLights.size());
    for (size_t i = 0; i < res.spotLights.size(); ++i) {
        slGPU[i].position = glm::vec4(res.spotLights[i].position, 0.0f);
        slGPU[i].direction = glm::vec4(glm::normalize(res.spotLights[i].direction), 0.0f);
        slGPU[i].color = glm::vec4(res.spotLights[i].color, res.spotLights[i].intensity);
        slGPU[i].angles = glm::vec4(
            cosf(glm::radians(res.spotLights[i].innerAngleDeg)),
            cosf(glm::radians(res.spotLights[i].outerAngleDeg)),
            0.0f, 0.0f);
    }
    if (!slGPU.empty()) {
        std::memcpy(res.spotLightSSBOMapped, slGPU.data(), slGPU.size() * sizeof(SpotLight));
    }

    auto updateObjectTransform = [&](ObjectState& obj) {
        glm::vec3 pos = obj.position;
        glm::vec3 rot = obj.rotationDeg;

        if (obj.animPlaying) {
            obj.animTime += dt * obj.animSpeed;
            const float t = static_cast<float>(obj.animTime);
            pos.x += obj.animRadius * std::cos(t);
            pos.z += obj.animRadius * std::sin(t);
            pos.y += 0.5f * std::sin(2.0f * t);
            rot.y += glm::degrees(t);
        }

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, pos);
        model = glm::rotate(model, glm::radians(rot.x), glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, glm::radians(rot.y), glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, glm::radians(rot.z), glm::vec3(0.0f, 0.0f, 1.0f));
        model = glm::scale(model, obj.scale);

        obj.ubo.model = model;
        obj.ubo.normalMatrix = glm::transpose(glm::inverse(model));
        obj.ubo.albedo = glm::vec4(obj.albedo, 1.0f);
        obj.ubo.specularColor = glm::vec4(obj.specularColor, 1.0f);
        obj.ubo.shininess = obj.shininess;
        obj.ubo.useLighting = obj.useLighting ? 1 : 0;
    };

    for (auto& obj : res.objects) {
        updateObjectTransform(obj);
    }

    if (!res.gizmoObjects.empty()) {
        vkQueueWaitIdle(ctx.graphics_queue);
        for (auto& g : res.gizmoObjects) {
            destroyObject(g);
        }
        res.gizmoObjects.clear();
    }

    if (res.showLightGizmos) {
        for (size_t i = 0; i < res.pointLights.size(); ++i) {
            ObjectState gizmo;
            gizmo.position = res.pointLights[i].position;
            gizmo.scale = glm::vec3(0.2f);
            gizmo.albedo = res.pointLights[i].color;
            gizmo.useLighting = false;
            if (createObject(gizmo, gizmo.position, gizmo.albedo)) {
                updateObjectTransform(gizmo);
                res.gizmoObjects.push_back(gizmo);
            }
        }

        for (size_t i = 0; i < res.spotLights.size(); ++i) {
            ObjectState gizmo;
            gizmo.position = res.spotLights[i].position;
            gizmo.scale = glm::vec3(0.2f);
            gizmo.albedo = res.spotLights[i].color;
            gizmo.useLighting = false;
            if (createObject(gizmo, gizmo.position, gizmo.albedo)) {
                updateObjectTransform(gizmo);
                res.gizmoObjects.push_back(gizmo);
            }

            ObjectState cone;
            glm::vec3 dir = glm::normalize(res.spotLights[i].direction);
            cone.position = res.spotLights[i].position;
            cone.albedo = res.spotLights[i].color;
            cone.useLighting = false;
            cone.scale = glm::vec3(1.0f);

            glm::vec3 up(0, 1, 0);
            if (fabsf(glm::dot(dir, up)) > 0.99f) up = glm::vec3(1, 0, 0);
            glm::vec3 right = glm::normalize(glm::cross(up, dir));
            glm::vec3 corrUp = glm::cross(dir, right);

            glm::mat4 rotMat(1.0f);
            rotMat[0] = glm::vec4(right, 0);
            rotMat[1] = glm::vec4(corrUp, 0);
            rotMat[2] = glm::vec4(dir, 0);
            rotMat[3] = glm::vec4(0, 0, 0, 1);

            if (createObject(cone, cone.position, cone.albedo)) {
                float coneScale = tanf(glm::radians(res.spotLights[i].outerAngleDeg)) * 1.5f;
                glm::mat4 model = glm::translate(glm::mat4(1.0f), cone.position) * rotMat *
                                  glm::scale(glm::mat4(1.0f), glm::vec3(coneScale, 1.5f, coneScale));

                cone.ubo.model = model;
                cone.ubo.normalMatrix = glm::transpose(glm::inverse(model));
                cone.ubo.albedo = glm::vec4(cone.albedo, 1.0f);
                cone.ubo.specularColor = glm::vec4(0.0f);
                cone.ubo.shininess = 1.0f;
                cone.ubo.useLighting = 0;
                res.gizmoObjects.push_back(cone);
            }
        }
    }
}

void render(const graphics::internal::FrameData& fd) {
    auto& ctx = graphics::internal::context;

    std::memcpy(res.sceneUniformMapped, &res.sceneUbo, sizeof(res.sceneUbo));

    for (auto& obj : res.objects) {
        std::memcpy(obj.uniformMapped, &obj.ubo, sizeof(obj.ubo));
    }
    for (auto& g : res.gizmoObjects) {
        std::memcpy(g.uniformMapped, &g.ubo, sizeof(g.ubo));
    }

    vkResetCommandBuffer(fd.command_buffer, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(fd.command_buffer, &beginInfo);

    const VkClearValue clearValues[2] = {
        VkClearValue{.color = {{0.02f, 0.02f, 0.03f, 1.0f}}},
        VkClearValue{.depthStencil = {1.0f, 0}},
    };
    VkRenderPassBeginInfo rpInfo{};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpInfo.renderPass = ctx.render_pass;
    rpInfo.framebuffer = fd.framebuffer;
    rpInfo.renderArea = {{0, 0}, ctx.swapchain_extent};
    rpInfo.clearValueCount = 2;
    rpInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(fd.command_buffer, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, res.pipeline);

    const VkViewport viewport = { 0.0f, 0.0f,
                                  float(ctx.swapchain_extent.width),
                                  float(ctx.swapchain_extent.height),
                                  0.0f, 1.0f };
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    const VkRect2D scissor = {{0, 0}, ctx.swapchain_extent};
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            res.pipelineLayout, 0, 1, &res.sceneDescriptorSet, 0, nullptr);

    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            res.pipelineLayout, 1, 1, &res.objects[0].descriptorSet, 0, nullptr);
    drawMesh(fd.command_buffer, res.boxMesh);

    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            res.pipelineLayout, 1, 1, &res.objects[1].descriptorSet, 0, nullptr);
    drawMesh(fd.command_buffer, res.boxMesh);

    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            res.pipelineLayout, 1, 1, &res.objects[2].descriptorSet, 0, nullptr);
    drawMesh(fd.command_buffer, res.floorMesh);

    if (res.objects.size() > 3) {
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                res.pipelineLayout, 1, 1, &res.objects[3].descriptorSet, 0, nullptr);
        drawMesh(fd.command_buffer, res.postMesh);
    }

    for (size_t gi = 0; gi < res.gizmoObjects.size(); ++gi) {
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                res.pipelineLayout, 1, 1, &res.gizmoObjects[gi].descriptorSet, 0, nullptr);

        bool isCone = false;
        if (gi >= res.pointLights.size() && ((gi - res.pointLights.size()) % 2 == 1)) {
            isCone = true;
        }

        if (isCone) {
            drawMesh(fd.command_buffer, res.coneMesh);
        } else {
            drawMesh(fd.command_buffer, res.boxMesh);
        }
    }

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);

    for (auto& g : res.gizmoObjects) {
        destroyObject(g);
    }

    for (auto& obj : res.objects) {
        destroyObject(obj);
    }

    vkDestroyPipeline(ctx.device, res.pipeline, nullptr);
    vkDestroyPipelineLayout(ctx.device, res.pipelineLayout, nullptr);
    vkDestroyDescriptorPool(ctx.device, res.descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, res.sceneSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, res.objectSetLayout, nullptr);

    if (res.pointLightSSBO != VK_NULL_HANDLE) {
        vmaDestroyBuffer(ctx.allocator, res.pointLightSSBO, res.pointLightSSBOAllocation);
    }
    if (res.spotLightSSBO != VK_NULL_HANDLE) {
        vmaDestroyBuffer(ctx.allocator, res.spotLightSSBO, res.spotLightSSBOAllocation);
    }

    destroyMesh(res.boxMesh);
    destroyMesh(res.floorMesh);
    destroyMesh(res.postMesh);
    destroyMesh(res.coneMesh);

    vmaDestroyBuffer(ctx.allocator, res.sceneUniformBuffer, res.sceneUniformAllocation);
}

} // namespace application