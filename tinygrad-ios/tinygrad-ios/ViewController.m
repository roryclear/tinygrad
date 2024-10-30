#import "ViewController.h"
#import <sys/socket.h>
#import <netinet/in.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <GZIP/GZIP.h>

@interface ViewController ()
@property (nonatomic) CFSocketRef socket;
@end

@implementation ViewController

NSMutableDictionary<NSString *, id> *objects;
id<MTLComputeCommandEncoder> encoder;
id<MTLDevice> device;
id<MTLCommandQueue> command_queue;
MTLComputePipelineDescriptor *desc;
NSMutableArray *queue;

- (void)viewDidLoad {
    device = MTLCreateSystemDefaultDevice();
    command_queue = [device newCommandQueueWithMaxCommandBufferCount: 1024 ];
    desc = [MTLComputePipelineDescriptor new];
    objects = [[NSMutableDictionary alloc] init];
    [desc setSupportIndirectCommandBuffers: true ];
    queue = [[NSMutableArray alloc] init];
    
    [super viewDidLoad];
    [self startHTTPServer];
}

uint8_t *convertNSStringToBytes(NSString *hexString) {
    uint8_t *bytes = malloc(4 * sizeof(uint8_t));
    NSArray<NSString *> *components = [hexString componentsSeparatedByString:@" "];
    for (NSInteger i = 0; i < components.count; i++) {
        unsigned int byteValue;
        [[NSScanner scannerWithString:components[i]] scanHexInt:&byteValue];
        bytes[i] = (uint8_t)byteValue;
    }
    return bytes;
}

char *charArrayFromMTLBuffer(id<MTLBuffer> buffer) {
    uint8_t *bytes = (uint8_t *)buffer.contents;
    NSUInteger length = buffer.length;
    char *hexString = malloc(length * 3);
    if (hexString == NULL) return NULL;
    char *p = hexString;
    for (NSUInteger i = 0; i < length; i++) {
        p += sprintf(p, "%02x", bytes[i]);
        if (i < length - 1) {
            *p++ = ' ';
        }
    }
    *p = '\0';
    return hexString;
}

- (void)startHTTPServer {
    self.socket = CFSocketCreate(NULL, PF_INET, SOCK_STREAM, IPPROTO_TCP, kCFSocketAcceptCallBack, AcceptCallback, NULL);
    if (!self.socket) {
        NSLog(@"Unable to create socket.");
        return;
    }
    
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_len = sizeof(address);
    address.sin_family = AF_INET;
    address.sin_port = htons(8081);  //use same port on tinygrad
    address.sin_addr.s_addr = INADDR_ANY;
    
    CFDataRef addressData = CFDataCreate(NULL, (const UInt8 *)&address, sizeof(address));
    if (CFSocketSetAddress(self.socket, addressData) != kCFSocketSuccess) {
        NSLog(@"Failed to bind socket to address.");
        CFRelease(self.socket);
        self.socket = NULL;
        exit(0); //TODO, add ui or retry
        return;
    }
    CFRelease(addressData);
    
    CFRunLoopSourceRef source = CFSocketCreateRunLoopSource(NULL, self.socket, 0);
    CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes);
    CFRelease(source);
    
    NSLog(@"HTTP Server started on port 8081.");
}

void printBufferBytes(id<MTLBuffer> buffer) {
    unsigned char *bytes = (unsigned char *)[buffer contents];
    NSUInteger length = [buffer length];
    NSMutableString *byteString = [NSMutableString stringWithCapacity:length * 3];
    for (NSUInteger i = 0; i < length; i++) {
        [byteString appendFormat:@"%02x ", bytes[i]];
    }
    NSLog(@"Buffer bytes: %@", byteString);
}

static void AcceptCallback(CFSocketRef socket, CFSocketCallBackType type, CFDataRef address, const void *data, void *info) {
    if (type != kCFSocketAcceptCallBack) return;
    
    CFSocketNativeHandle handle = *(CFSocketNativeHandle *)data;
    char buffer[1024*500] = {0}; //TODO how big/small should this be?
    
    ssize_t receivedBytes = recv(handle, buffer, sizeof(buffer) - 1, 0);
    if (receivedBytes < 1) {
        NSLog(@"Failed to receive data.");
        close(handle);
        return;
    }
    
    buffer[receivedBytes] = '\0';
    CFDataRef dataRef = CFDataCreate(NULL, (UInt8 *)buffer, (CFIndex)receivedBytes);
    CFHTTPMessageRef httpRequest = CFHTTPMessageCreateEmpty(NULL, TRUE);
    CFHTTPMessageAppendBytes(httpRequest, CFDataGetBytePtr(dataRef), CFDataGetLength(dataRef));
    
    if (CFHTTPMessageIsHeaderComplete(httpRequest)) {
        //NSData *bodyData = (__bridge_transfer NSData *)CFHTTPMessageCopyBody(httpRequest);
        NSData *bodyDataUnc = (__bridge_transfer NSData *)CFHTTPMessageCopyBody(httpRequest);
        NSData *bodyData = [bodyDataUnc gunzippedData];
        
        if (!bodyData) {
            NSLog(@"no body data");
            const char *response = "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\n\r\nInvalid request: Missing or malformed body.";
            send(handle, response, strlen(response), 0);
            close(handle);
            CFRelease(httpRequest);
            CFRelease(dataRef);
            return;
        }

        NSError *jsonError = nil;
        NSDictionary *jsonDict = [NSJSONSerialization JSONObjectWithData:bodyData options:0 error:&jsonError];
        
        if (!jsonDict || jsonError) {
            NSLog(@"no json error");
            const char *response = "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\n\r\nInvalid request: Missing or malformed body.";
            send(handle, response, strlen(response), 0);
            close(handle);
            CFRelease(httpRequest);
            CFRelease(dataRef);
            return;
        }
        
        NSArray *req_queue = jsonDict[@"queue"];
        [queue addObjectsFromArray:req_queue];
        if([[queue lastObject][0] isEqualToString:@"copyout"]) {
            for (int i = 0; i < [queue count]; i++) {
                if ([queue[i][0] isEqualToString:@"new_buffer"])  {
                    [objects setObject:[device newBufferWithLength:[queue[i][2] intValue] options:MTLResourceStorageModeShared] forKey:queue[i][1]];
                }
                if ([queue[i][0] isEqualToString:@"memcpy"])  {
                    if ([objects objectForKey:queue[i][2]] == nil) {
                        [objects setObject:[NSData dataWithContentsOfURL:[[NSBundle mainBundle] URLForResource:queue[i][2] withExtension:nil]] forKey:queue[i][2]];
                    }
                    memcpy([(id<MTLBuffer>)objects[queue[i][1]] contents] + 0, [(NSData *)objects[queue[i][2]] bytes] + [queue[i][3] intValue], [queue[i][4] intValue]); //TODO check this, also dest offset?
                }
                if ([queue[i][0] isEqualToString:@"copyout"])  {
                    char *bytes = charArrayFromMTLBuffer(objects[queue[i][1]]);
                    char *response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n";
                    size_t totalLength = strlen(response) + strlen(bytes) + 1;
                    char *fullResponse = malloc(totalLength);
                    strcpy(fullResponse, response);
                    strcat(fullResponse, bytes);
                    send(handle, fullResponse, strlen(fullResponse), 0);
                    close(handle);
                    [queue removeAllObjects];
                    return;
                }
                if ([queue[i][0] isEqualToString:@"new_function"])  {
                    [objects setObject:[objects[queue[i][1]] newFunctionWithName: queue[i][1]] forKey:queue[i][2]]; //TODO don't need this structure?
                }
                if ([queue[i][0] isEqualToString:@"new_pipeline_state"])  {
                    [desc setComputeFunction: objects[queue[i][1]]];
                    NSError *error = nil;
                    [objects setObject:[device newComputePipelineStateWithDescriptor: desc options: 0 reflection: Nil error: &error ] forKey:queue[i][2]];
                }
                if ([queue[i][0] isEqualToString:@"new_command_buffer"]) { //TODO this freezes it, need to clear them?
                    [objects setObject:[command_queue commandBuffer] forKey:queue[i][1]];
                }
                if ([queue[i][0] isEqualToString:@"set_encoder"]) {
                    encoder = [objects[queue[i][1]] computeCommandEncoder]; //do I need to init this?
                }
                if ([queue[i][0] isEqualToString:@"set_pipeline_state"]) {
                    [encoder setComputePipelineState: objects[queue[i][1]]];
                }
                if ([queue[i][0] isEqualToString:@"set_buffer"]) {
                    [encoder setBuffer: objects[queue[i][1]] offset: [queue[i][2] intValue] atIndex: [queue[i][3] intValue] ];
                }
                if ([queue[i][0] isEqualToString:@"set_bytes"]) {
                    [encoder setBytes: convertNSStringToBytes(queue[i][1]) length: 4 atIndex: [queue[i][3] intValue] ];
                }
                if ([queue[i][0] isEqualToString:@"dispatch"]) {
                    [encoder dispatchThreadgroups: MTLSizeMake([queue[i][1] intValue], [queue[i][2] intValue], [queue[i][3] intValue]) threadsPerThreadgroup: MTLSizeMake([queue[i][4] intValue], [queue[i][5] intValue], [queue[i][6] intValue]) ];
                    [encoder endEncoding];
                }
                if ([queue[i][0] isEqualToString:@"commit"]) {
                    [objects[queue[i][1]] commit];
                }
                if ([queue[i][0] isEqualToString:@"wait"]) {
                    [objects[queue[i][1]] waitUntilCompleted];
                    [objects removeObjectForKey:queue[i][1]];
                }
                if ([queue[i][0] isEqualToString:@"new_library"]) {
                    [objects setObject:[device newLibraryWithSource:queue[i][1] options:nil error:nil] forKey:queue[i][2]];
                }
                if ([queue[i][0] isEqualToString:@"copy_in"]) {
                    NSArray<NSString *> *hexArray = [queue[i][1] componentsSeparatedByString:@" "];
                    NSUInteger length = hexArray.count;
                    uint8_t *bytes = malloc(length);
                    for (NSUInteger i = 0; i < length; i++) {
                        unsigned int byteValue;
                        [[NSScanner scannerWithString:hexArray[i]] scanHexInt:&byteValue];
                        bytes[i] = (uint8_t)byteValue;
                    }
                    NSData *data = [NSData dataWithBytesNoCopy:bytes length:length freeWhenDone:YES];
                    memcpy([(id<MTLBuffer>)objects[queue[i][2]] contents], [data bytes], [data length]);
                }
            }
        }
    }
    
    const char *response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n";
    send(handle, response, strlen(response), 0);
    close(handle);
}

@end
