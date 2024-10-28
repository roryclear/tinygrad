#import "ViewController.h"
#import <sys/socket.h>
#import <netinet/in.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface ViewController ()
@property (nonatomic) CFSocketRef socket;
@end

@implementation ViewController

NSMutableDictionary<NSString *, id<MTLBuffer>> *buffers;
NSMutableDictionary<NSString *, NSData *> *file_data;
NSMutableDictionary<NSString *, id<MTLFunction>> *functions;
NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipeline_states;
id<MTLDevice> device;
id<MTLLibrary> library; //TODO use string, instead of file?
MTLComputePipelineDescriptor *desc;

- (void)viewDidLoad {
    device = MTLCreateSystemDefaultDevice();
    buffers = [[NSMutableDictionary alloc] init];
    functions = [[NSMutableDictionary alloc] init];
    file_data = [[NSMutableDictionary alloc] init];
    pipeline_states = [[NSMutableDictionary alloc] init];
    library = [MTLCreateSystemDefaultDevice() newDefaultLibrary];
    desc = [MTLComputePipelineDescriptor new];
    [desc setSupportIndirectCommandBuffers: true ];
    
    //NSData *f113965bb5fe7074edc9ca25991e7ad35 = [NSData dataWithContentsOfURL:[[NSBundle mainBundle] URLForResource:@"113965bb5fe7074edc9ca25991e7ad35" withExtension:nil]]; //TODO
    [file_data setObject:[NSData dataWithContentsOfURL:[[NSBundle mainBundle] URLForResource:@"113965bb5fe7074edc9ca25991e7ad35" withExtension:nil]] forKey:@"113965bb5fe7074edc9ca25991e7ad35"]; //TODO
    [super viewDidLoad];
    [self startHTTPServer];
}

- (void)new_buffer:(NSString *)key size:(int)size {
    [buffers setObject:[device newBufferWithLength:size options:MTLResourceStorageModeShared] forKey:key];
}

- (void)startHTTPServer {
    // Create a socket
    self.socket = CFSocketCreate(NULL, PF_INET, SOCK_STREAM, IPPROTO_TCP, kCFSocketAcceptCallBack, AcceptCallback, NULL);
    if (!self.socket) {
        NSLog(@"Unable to create socket.");
        return;
    }
    
    // Set up socket address and port
    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_len = sizeof(address);
    address.sin_family = AF_INET;
    address.sin_port = htons(8081);  // Port 8080
    address.sin_addr.s_addr = INADDR_ANY;
    
    CFDataRef addressData = CFDataCreate(NULL, (const UInt8 *)&address, sizeof(address));
    if (CFSocketSetAddress(self.socket, addressData) != kCFSocketSuccess) {
        NSLog(@"Failed to bind socket to address.");
        CFRelease(self.socket);
        self.socket = NULL;
        return;
    }
    CFRelease(addressData);
    
    // Create a run loop source and add to current run loop
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

// Callback function to handle incoming connections
static void AcceptCallback(CFSocketRef socket, CFSocketCallBackType type, CFDataRef address, const void *data, void *info) {
    if (type != kCFSocketAcceptCallBack) return;
    
    // Accept the incoming connection
    CFSocketNativeHandle handle = *(CFSocketNativeHandle *)data;
    char buffer[1024*100] = {0}; //TODO how big/small should this be?
    
    // Read data from the client
    ssize_t receivedBytes = recv(handle, buffer, sizeof(buffer) - 1, 0);
    if (receivedBytes < 1) {
        NSLog(@"Failed to receive data.");
        close(handle);
        return;
    }
    
    // Null-terminate and log received data
    buffer[receivedBytes] = '\0';
    NSLog(@"Received data: %s", buffer);

    // Create CFData from the received buffer
    CFDataRef dataRef = CFDataCreate(NULL, (UInt8 *)buffer, (CFIndex)receivedBytes);
    CFHTTPMessageRef httpRequest = CFHTTPMessageCreateEmpty(NULL, TRUE);
    CFHTTPMessageAppendBytes(httpRequest, CFDataGetBytePtr(dataRef), CFDataGetLength(dataRef));

    // Check if it's a complete HTTP request
    if (CFHTTPMessageIsHeaderComplete(httpRequest)) {
        // Extract the JSON body from the HTTP message
        NSData *bodyData = (__bridge_transfer NSData *)CFHTTPMessageCopyBody(httpRequest);
        
        if (!bodyData) {
            // Respond with 400 Bad Request
            const char *response = "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\n\r\nInvalid request: Missing or malformed body.";
            send(handle, response, strlen(response), 0);
            
            // Clean up and return
            close(handle);
            CFRelease(httpRequest);
            CFRelease(dataRef);
            return;
        }
        
        // Parse the JSON data
        NSError *jsonError = nil;
        NSDictionary *jsonDict = [NSJSONSerialization JSONObjectWithData:bodyData options:0 error:&jsonError];
        
        if (!jsonDict || jsonError) {
            NSLog(@"Failed to parse JSON: %@", jsonError);
            close(handle);
            CFRelease(httpRequest);
            CFRelease(dataRef);
            return;
        }
        
        // Log the received JSON dictionary
        //NSLog(@"Received JSON: %@", jsonDict);
        NSArray *queue = jsonDict[@"queue"];
        for (int i = 0; i < [queue count]; i++) {
            NSLog(@"Element at index %d: %@", i, queue[i]);
            if ([queue[0] count] > 0) {
                NSLog(@"func? %@",queue[i][0]);
                if ([queue[i][0] isEqualToString:@"new_buffer"])  {
                    [buffers setObject:[device newBufferWithLength:[queue[i][2] intValue] options:MTLResourceStorageModeShared] forKey:queue[i][1]];
                }
                if ([queue[i][0] isEqualToString:@"memcpy"])  {
                    memcpy([buffers[queue[i][1]] contents] + 0, [file_data[queue[i][2]] bytes] + [queue[i][3] intValue], [queue[i][4] intValue]); //TODO check this, also dest offset?
                }
                if ([queue[i][0] isEqualToString:@"copyout"])  {
                    printBufferBytes(buffers[queue[i][1]]);
                }
                if ([queue[i][0] isEqualToString:@"new_function"])  {
                    [functions setObject:[library newFunctionWithName: queue[i][1]] forKey:queue[i][2]];
                }
                if ([queue[i][0] isEqualToString:@"new_pipeline_state"])  {
                    [desc setComputeFunction: functions[queue[i][1]]];
                    NSError *error = nil;
                    [pipeline_states setObject:[device newComputePipelineStateWithDescriptor: desc options: 0 reflection: Nil error: &error ] forKey:queue[i][2]];
                }
                
            }
        }
        
        
    }
    
    // Simple response for any request
    const char *response = "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nHello from iPhone!";
    send(handle, response, strlen(response), 0);
    close(handle);
}

@end
