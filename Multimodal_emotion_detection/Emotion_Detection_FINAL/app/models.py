# app/models.py

import tensorflow as tf
import numpy as np

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention = tf.keras.layers.Dense(input_shape[-1], activation='softmax')
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention_weights = self.attention(inputs)
        return tf.multiply(inputs, attention_weights)

    def get_config(self):
        return super(AttentionLayer, self).get_config()

class EmotionModels:
    def __init__(self, models_dir='models'):
        # Define custom objects
        custom_objects = {'AttentionLayer': AttentionLayer}
        
        # Load models with custom objects
        with tf.keras.utils.custom_object_scope(custom_objects):
            try:
                # Try loading the full model first
                self.multimodal_model = tf.keras.models.load_model(
                    f'{models_dir}/multimodal_model_enhanced (1).h5')
            except:
                # Fallback: Build model architecture and load weights separately
                print("Warning: Couldn't load full model, attempting weights-only load")
                self.multimodal_model = self._build_multimodal_model()
                self.multimodal_model.load_weights(
                    f'{models_dir}/multimodal_model_enhanced (1).h5')
            
            # Load other models
            self.audio_model = tf.keras.models.load_model(
                f'{models_dir}/audio_model_enhanced (2).h5')
            self.video_model = tf.keras.models.load_model(
                f'{models_dir}/best_video_model (1).h5')

        # Load label encoders
        self.audio_labels = np.load(f'{models_dir}/audio_label_encoder_enhanced (1).npy', allow_pickle=True)
        self.video_labels = np.load(f'{models_dir}/video_label_encoder_enhanced (1).npy', allow_pickle=True)
        self.multimodal_labels = np.load(f'{models_dir}/multimodal_label_encoder_enhanced (1).npy', allow_pickle=True)

    def _build_multimodal_model(self):
        """Reconstruct the exact model architecture"""
        # Input layers
        audio_input = tf.keras.Input(shape=(None,))  # Adjust shape as needed
        video_input = tf.keras.Input(shape=(128, 128, 3))
        
        # Dummy submodels (these will be replaced with loaded weights)
        audio_features = tf.keras.layers.Dense(64)(audio_input)
        video_features = tf.keras.layers.Flatten()(video_input)
        
        # Combined architecture
        combined = tf.keras.layers.Concatenate()([audio_features, video_features])
        combined = AttentionLayer()(combined)
        
        x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(combined)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        output = tf.keras.layers.Dense(len(self.multimodal_labels), activation='softmax')(x)
        
        return tf.keras.Model(inputs=[audio_input, video_input], outputs=output)
    
    def predict_audio(self, audio_features):
        pred = self.audio_model.predict(np.expand_dims(audio_features, axis=0))
        idx = np.argmax(pred)
        return self.audio_labels[idx], pred[0][idx]
    
    def predict_video(self, frame):
        frame = np.expand_dims(frame, axis=0)
        pred = self.video_model.predict(frame)
        idx = np.argmax(pred)
        return self.video_labels[idx], pred[0][idx]
    
    def predict_multimodal(self, audio_features, frame):
        audio_input = np.expand_dims(audio_features, axis=0)
        frame_input = np.expand_dims(frame, axis=0)
        pred = self.multimodal_model.predict([audio_input, frame_input])
        idx = np.argmax(pred)
        return self.multimodal_labels[idx], pred[0][idx]