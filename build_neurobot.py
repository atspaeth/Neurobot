from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef("""
        void setup();
        void cleanup();
        volatile extern bool g_please_die_kthxbai;
        extern int g_dt_us;
        float dt_ms();
        float read_adc(int channel_index);
        void apply_actuator(int i, float signed_fractional_activation);
        float get_current_time();
        void synchronize_loop();
        void print_final_time();
        void set_pwm_max(float percent);
""")

ffibuilder.set_source('neurobot_cffi', 
        """
        #include "libneurobot.h"
        """,
    libraries=['rt', 'pruio', 'neurobot'],
    library_dirs=['.'])

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
