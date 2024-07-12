'''from fg_auto_pipeline import both_data, separate_unique_data

from api import get_admin_ids ,get_event_ids , get_event_and_user_image, post_data


def processing_to_analyze_and_post(data):
    user_images=data['UserImages'][0]['image']
    print(len(data['EventImages']), len(user_images))
    if data is not None:
        similar_event_faces=both_data(data['EventImages'], data['UserImages'])
        # print(similar_event_faces)
        unique_data=separate_unique_data(similar_event_faces)
        print(unique_data)
        # post_data(unique_data)

def wholetrigger():
    admin_ids = get_admin_ids()
    if admin_ids:
        for admin_id in admin_ids:
            event_ids = get_event_ids(admin_id)
            print(event_ids)
            if event_ids:
                for event_id in event_ids:
                    data = get_event_and_user_image(admin_id, event_id)
                    processing_to_analyze_and_post(data)'''


from fg_auto_pipeline import FaceRecognitionPipeline
from api import APIClient

class FaceRecognitionService:
    def __init__(self, base_url):
        self.api_client = APIClient(base_url)
        self.pipeline = FaceRecognitionPipeline()

    def processing_to_analyze_and_post(self, data):
        if data is not None:
            similar_event_faces = self.pipeline.both_data(data['EventImages'], data['UserImages'])
            unique_data = self.pipeline.separate_unique_data(similar_event_faces)
            print(unique_data)
        
        return unique_data
    
    def whole_trigger(self):
        admin_ids = self.api_client.get_admin_ids()
        print(admin_ids)
        if admin_ids:
            for admin_id in admin_ids:
                # print(admin_id)
                event_ids = self.api_client.get_event_ids(admin_id)
                print(event_ids)
                if event_ids:
                    whole_events_unique_data=[]
                    for event_id in event_ids:
                        #To select a particular event
                        if event_id==2:
                            print('data is preparing...')
                            data = self.api_client.get_event_and_user_image(admin_id, event_id)
                            print(data)
                            to_get_one_event_unique_data=self.processing_to_analyze_and_post(data)
                            whole_events_unique_data.extend(to_get_one_event_unique_data)

                    # self.api_client.post_data(whole_events_unique_data)  

