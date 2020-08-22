#include <stdlib.h>                                      // Tag.OTHER
int main()                                               // Tag.OTHER
{                                                        // Tag.OTHER
    int entity_8;                                        // Tag.BODY
    int entity_3;                                        // Tag.BODY
    char entity_5[40];                                   // Tag.BODY
    entity_8 = 73;                                       // Tag.BODY
    char entity_1[70];                                   // Tag.BODY
    int entity_6;                                        // Tag.BODY
    entity_3 = 69;                                       // Tag.BODY
    for(entity_6 = 42; entity_6 < entity_3; entity_6++){ // Tag.BODY
    entity_5[entity_8] = 'f';                            // Tag.BUFWRITE_TAUT_UNSAFE
    }                                                    // Tag.BODY
    entity_1[entity_6] = 'd';                            // Tag.BUFWRITE_COND_SAFE
    return 0;                                            // Tag.BODY
}                                                        // Tag.OTHER