<script setup lang="ts">
import { ref } from 'vue';
import Slider from 'primevue/slider';

const frame_stride = ref(4);
const attention_window = ref(4);
const freeze_ratio = ref(0.6);
</script>

<template>
  <div class="node">
    <h3>LTX2 Efficient Sampler</h3>

    <div class="control">
        <label>Frame Stride: {{ frame_stride }}</label>
        <Slider v-model="frame_stride" :min="1" :max="8" />
    </div>

    <div class="control">
        <label>Temporal Window: {{ attention_window }}</label>
        <Slider v-model="attention_window" :min="1" :max="16" />
    </div>

    <div class="control">
        <label>Freeze Spatial @ {{ (freeze_ratio * 100).toFixed(0) }}%</label>
        <Slider v-model="freeze_ratio" :min="0" :max="1" :step="0.05" />
    </div>
  </div>
</template>

<style scoped>
.node {
    background: #2a2a2a;
    padding: 1rem;
    border-radius: 8px;
    color: white;
}
.control {
    margin-bottom: 1rem;
}
label {
    display: block;
    margin-bottom: 0.5rem;
}
</style>
