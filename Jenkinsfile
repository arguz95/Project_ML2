node {

    checkout scm

    docker.withRegistry('https://registry.hub.docker.com', 'dockerhub') {

        def customImage = docker.build("Ram1633/Project_ML2/Docker_Test")

        /* Push the container to the custom Registry */
        customImage.push()
    }
}